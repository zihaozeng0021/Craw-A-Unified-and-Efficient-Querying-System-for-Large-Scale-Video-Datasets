import argparse
import os

from loader.image_loader import ImageLoader
from predicate.count_predicate import CountPredicate
from evaluator.f1_evaluator import F1Evaluator
from naive.scheduler import Scheduler as NaiveScheduler
from msfilter.scheduler import Scheduler as MSFilterScheduler
from mecoarse.scheduler import Scheduler as MECoarseScheduler
from figo.scheduler import Scheduler as FiGOScheduler
from mc.scheduler import Scheduler as MCScheduler
from profiler.model_time_titanx import setup_static_model_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sys", required=True, type=str)
    parser.add_argument("--modeling", required=True, type=str)
    parser.add_argument("--use-cache", action="store_true", default=False)
    parser.add_argument(
        "--use-titanx-model-time", action="store_true", default=False
    )
    args = parser.parse_args()

    pred = CountPredicate(["car"], [4])

    loader = ImageLoader("ua-detrac", args.use_cache)

    if args.use_titanx_model_time:
        setup_static_model_time()

    if args.sys == "naive":
        sched_f = NaiveScheduler
    elif args.sys == "ms-filter":
        sched_f = MSFilterScheduler
    elif args.sys == "me-coarse":
        sched_f = MECoarseScheduler
    elif args.sys == "figo":
        sched_f = FiGOScheduler
    elif args.sys == "mc":
        sched_f = MCScheduler
    else:
        raise Exception("System {} is not defined".format(args.sys))

    ref_sched = NaiveScheduler(args.modeling, loader, pred, args.use_cache)
    ref_res = ref_sched.process()

    f1, query_time, opt_time = 0, 0, 0
    for _ in range(5):
        sched = sched_f(args.modeling, loader, pred, args.use_cache)
        res = sched.process()

        evaluator = F1Evaluator(pred)
        f1 += evaluator.evaluate(res, ref_res)
        query_time += sched.get_query_time()
        opt_time += sched.get_optimization_time()

    f1 /= 5
    query_time /= 5
    opt_time /= 5

    print("Query time: {:.3f} sec".format(query_time))
    print("F1: {:.3f}".format(f1))

    with open("./res/q1/res.txt", "a+") as f:
        f.write(
            "{},{},{},{}\n".format(
                f1,
                query_time,
                opt_time,
                query_time - opt_time,
            )
        )
    f.close()


if __name__ == "__main__":
    main()
