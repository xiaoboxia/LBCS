import abc
import tqdm
import time
import pathlib

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, tqdm_writer=True):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if not tqdm_writer:
            print("\t".join(entries))
        else:
            tqdm.tqdm.write("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits+2) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class Meter(object):
    @abc.abstractmethod
    def __init__(self, name, fmt=":f"):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, val, n=1):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


class AverageMeter(Meter):
    """ Computes and stores the average and current value """

    def __init__(self, name, fmt=":f", write_val=True, write_avg=True):
        self.name = name
        self.fmt = fmt
        self.reset()

        self.write_val = write_val
        self.write_avg = write_avg

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class VarianceMeter(Meter):
    def __init__(self, name, fmt=":f", write_val=False):
        self.name = name
        self._ex_sq = AverageMeter(name="_subvariance_1", fmt=":.02f")
        self._sq_ex = AverageMeter(name="_subvariance_2", fmt=":.02f")
        self.fmt = fmt
        self.reset()
        self.write_val = False
        self.write_avg = True

    @property
    def val(self):
        return self._ex_sq.val - self._sq_ex.val ** 2

    @property
    def avg(self):
        return self._ex_sq.avg - self._sq_ex.avg ** 2

    def reset(self):
        self._ex_sq.reset()
        self._sq_ex.reset()

    def update(self, val, n=1):
        self._ex_sq.update(val ** 2, n=n)
        self._sq_ex.update(val, n=n)

    def __str__(self):
        return ("{name} (var {avg" + self.fmt + "})").format(
            name=self.name, avg=self.avg
        )

def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Epoch, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5, "
            "Setting\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{epoch}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}, "
                "{setting}\n"
            ).format(now=now, **kwargs)
        )