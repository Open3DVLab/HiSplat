import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerBoundedCfg:
    name: Literal["bounded", "bounded_dtu"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    special_way: str | bool | None


class ViewSamplerBounded(ViewSampler[ViewSamplerBoundedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        num_views, _, _ = extrinsics.shape

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.max_distance_between_context_views
        elif self.cfg.warm_up_steps > 0:
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
            )
        else:
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views

        # Pick the gap between the context views.
        # NOTE: we keep the bug untouched to follow initial pixelsplat cfgs
        if self.cfg.special_way is None:
            if not self.cameras_are_circular:
                max_gap = min(num_views - 1, min_gap)
            min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        elif self.cfg.special_way == "debug_gap":
            if not self.cameras_are_circular:
                max_gap = min(num_views - 1, max_gap)
            min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        elif self.cfg.special_way == "random":
            max_gap = num_views - 1
            min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        else:
            raise NotImplementedError(f"{self.cfg.special_way} is Not implemented!")

        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames!")
        context_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        # Pick the left and right context indices.
        index_context_left = torch.randint(
            num_views if self.cameras_are_circular else num_views - context_gap,
            size=tuple(),
            device=device,
        ).item()
        if self.stage == "test":
            index_context_left = index_context_left * 0
        index_context_right = index_context_left + context_gap

        if self.is_overfitting:
            index_context_left *= 0
            index_context_right *= 0
            index_context_right += max_gap

        # Pick the target view indices.
        if self.stage == "test":
            # When testing, pick all.
            index_target = torch.arange(
                index_context_left,
                index_context_right + 1,
                device=device,
            )
        else:
            # When training or validating (visualizing), pick at random.
            index_target = torch.randint(
                index_context_left + self.cfg.min_distance_to_context_views,
                index_context_right + 1 - self.cfg.min_distance_to_context_views,
                size=(self.cfg.num_target_views,),
                device=device,
            )

        # Apply modulo for circular datasets.
        if self.cameras_are_circular:
            index_target %= num_views
            index_context_right %= num_views

        return (
            torch.tensor((index_context_left, index_context_right)),
            index_target,
        )

    @property
    def num_context_views(self) -> int:
        return 2

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views


class ViewSamplerBoundedDTU(ViewSampler[ViewSamplerBoundedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    # TODO: we only consider the train time but no implement for test time
    # a simple sampling way for 2-view DTU
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        self.matrix_w = 6
        self.matrix_h = 8
        num_views, _, _ = extrinsics.shape
        # use matrix distance to determine near or not near
        # Compute the context view spacing based on the current global step.
        if self.cfg.warm_up_steps > 0:
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
            )
        else:
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views

        # Pick the gap between the context views.
        # NOTE: we keep the bug untouched to follow initial pixelsplat cfgs
        # TSJ: we debug and apply the max gap
        max_gap = min(num_views - 1, max_gap)
        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames!")

        context_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()
        index_context_left, index_context_right, index_target = self.get_context_target_coordinate(
            context_gap, self.cfg.num_target_views
        )
        index_target = torch.tensor(index_target, device=device)
        return (
            torch.tensor((index_context_left, index_context_right)),
            index_target,
        )

    @property
    def num_context_views(self) -> int:
        return 2

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views

    def get_context_target_coordinate(self, context_gap, target_num):
        assert (
            (context_gap >= (min(self.matrix_w, self.matrix_h) - 1))
            and (context_gap <= (self.matrix_w + self.matrix_h - 2))
            and (target_num < context_gap)
        )
        candidate_x = [
            k for k in range(self.matrix_w) if max(k, self.matrix_w - 1 - k) >= (context_gap - self.matrix_h + 1)
        ]
        context_a_x = np.random.choice(candidate_x, 1)[0]
        max_context_a_x = max(context_a_x, self.matrix_w - 1 - context_a_x)
        rest_a_max_y = max(0, context_gap - max_context_a_x)
        candidate_y = [k for k in range(self.matrix_h) if max(k, self.matrix_h - 1 - k) >= rest_a_max_y]
        context_a_y = np.random.choice(candidate_y, 1)[0]
        context_a_point = (context_a_x, context_a_y)
        context_a_index = context_a_x + context_a_y * self.matrix_w
        # sample b point
        rest_b_max_y = max(context_a_y, self.matrix_h - 1 - context_a_y)
        rest_b_max_x = max(0, context_gap - rest_b_max_y)
        candidate_x_b = [k for k in range(self.matrix_w) if abs(k - context_a_x) >= rest_b_max_x]
        context_b_x = np.random.choice(candidate_x_b, 1)[0]
        rest_b_y = context_gap - abs(context_a_x - context_b_x)
        if rest_b_y == 0:
            context_b_y = context_a_y
        else:
            if context_a_y + rest_b_y > self.matrix_h - 1:
                context_b_y = context_a_y - rest_b_y
            elif context_a_y - rest_b_y < 0:
                context_b_y = context_a_y + rest_b_y
            else:
                candidate_y_b = [context_a_y + rest_b_y, context_a_y - rest_b_y]
                context_b_y = np.random.choice(candidate_y_b, 1)[0]
        context_b_point = (context_b_x, context_b_y)
        context_b_index = context_b_x + context_b_y * self.matrix_w

        # sample target point
        target_candidate_x = list(range(*sorted([context_a_x, context_b_x]))) + [max(context_a_x, context_b_x)]
        target_candidate_y = list(range(*sorted([context_a_y, context_b_y]))) + [max(context_a_y, context_b_y)]
        target_candidate_point = list(itertools.product(target_candidate_x, target_candidate_y))
        target_candidate_point = list(
            filter(
                lambda x: not (
                    (x[0] == context_a_x and x[1] == context_a_y) or (x[0] == context_b_x and x[1] == context_b_y)
                ),
                target_candidate_point,
            )
        )
        target_candidate_point = np.array(target_candidate_point)
        target_chosen_index = np.random.choice(range(len(target_candidate_point)), target_num, replace=False)
        target_point_coordinate = target_candidate_point[target_chosen_index]
        target_point_index = [p[0] + p[1] * self.matrix_w for p in target_point_coordinate]
        return context_a_index, context_b_index, target_point_index
