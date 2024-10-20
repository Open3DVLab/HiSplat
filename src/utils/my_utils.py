import datetime


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_duration(seconds):
    duration = datetime.timedelta(seconds=seconds)
    days = duration.days
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0:
        parts.append(f"{seconds}s")

    if len(parts) == 0:
        return "0s"
    return ":".join(parts)


def filter_low_opacities(gaussians, threshold=0.01):
    gaussians.opacities[gaussians.opacities < threshold] = 0


def plot_and_save_hist(x, path):
    import matplotlib.pyplot as plt

    plt.cla()
    x = x.detach().cpu().numpy()
    plt.hist(x[0], bins=20, range=(0, 1))
    plt.savefig(path)
