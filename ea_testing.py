from pathlib import Path

from scipy.io import arff


ARFF_PATH = Path(r"C:\Users\eca4zm\Downloads\0001_beatinfo.arff")


def main() -> None:
    data, meta = arff.loadarff(ARFF_PATH)

    print(f"Loaded: {ARFF_PATH}")
    print(f"Rows: {len(data)}")
    print(f"Columns: {meta.names()}")
    print(data[:5])


if __name__ == "__main__":
    main()
