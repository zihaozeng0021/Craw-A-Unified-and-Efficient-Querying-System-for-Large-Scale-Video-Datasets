import os


def main():
    # efficientdet
    base_url = (
        "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/"
        "releases/download/1.0/efficientdet-d{}.pth"
    )

    os.chdir("efficientdet")
    for i in range(8):
        os.system("wget " + base_url.format(i))


if __name__ == "__main__":
    main()
