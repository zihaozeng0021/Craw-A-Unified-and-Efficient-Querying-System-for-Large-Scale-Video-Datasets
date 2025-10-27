import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    args = parser.parse_args()

    per_col_total = []
    count = 0

    with open(args.path) as f:
        for line in f.readlines():
            count += 1
            line_arr = line.split(",")

            if len(per_col_total) == 0:
                for _ in range(len(line_arr)):
                    per_col_total.append(0)
            else:
                for i, num in enumerate(line_arr):
                    per_col_total[i] += float(num)

    print(",".join("{:.3f}".format(num / count) for num in per_col_total))


if __name__ == "__main__":
    main()
