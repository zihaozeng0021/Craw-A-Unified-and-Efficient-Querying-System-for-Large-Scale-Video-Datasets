import os
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    root_dir = args.cache_path

    sys_list = ["figo", "ms-filter", "me-coarse", "mc"]

    if args.query == "q1":
        dataset = "ua-detrac"
    elif args.query == "q2":
        dataset = "bdd"
    elif args.query == "q3":
        dataset = "virat"
    elif args.query == "q4":
        dataset = "jackson"
    elif args.query == "q5":
        dataset = "ua-detrac"
    elif args.query == "q6":
        dataset = "bdd"
    else:
        raise Exception("Query {} is not defined.".format(args.query))

    for sys in sys_list:
        # remove previous results
        if os.path.exists(os.path.join("./res", args.query, "res.txt")):
            os.remove(os.path.join("./res", args.query, "res.txt"))

        for path in os.listdir(os.path.join(root_dir, dataset)):
            print("-------- {} --------".format(path))
            cache_res_path = os.path.join(root_dir, dataset, path)

            if os.path.exists(os.path.join("./cache", dataset)):
                os.remove(os.path.join("./cache", dataset))

            os.system("ln -sf {} ./cache/{}".format(cache_res_path, dataset))
            os.system(
                "python query/{}.py --sys {} --modeling efficientdet --use-cache --use-titanx-model-time".format(
                    args.query, sys
                )
            )

        os.system(
            "cp {} {}".format(
                os.path.join("./res", args.query, "res.txt"),
                os.path.join("./res", args.query, "{}.txt".format(sys)),
            )
        )


if __name__ == "__main__":
    main()
