"""dunnhumby 공식 공개 ZIP에서 주요 CSV만 안전하게 추출합니다."""
from pathlib import Path
from urllib.request import urlretrieve
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import shutil

SOURCE_URL = "https://downloads.ctfassets.net/psj0p18eh7z1/3e9OAF7F9ONT4pwJc1luEw/d56af8aabad51bdb9888aad0240bd105/dunnhumby_The-Complete-Journey.zip"
FILES = {"transaction_data.csv", "product.csv", "hh_demographic.csv", "campaign_table.csv", "campaign_desc.csv"}


def main():
    destination = Path(__file__).resolve().parent / "data" / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="retail-data-") as temp:
        archive = Path(temp) / "source.zip"
        print("dunnhumby 공식 공개 ZIP을 내려받습니다.")
        urlretrieve(SOURCE_URL, archive)
        found = set()
        with ZipFile(archive) as source:
            for entry in source.infolist():
                name = Path(entry.filename).name
                if name in FILES:
                    with source.open(entry) as reader, (destination / name).open("wb") as writer:
                        shutil.copyfileobj(reader, writer)
                    found.add(name)
        if found != FILES:
            raise ValueError(f"공식 ZIP에 필요한 파일이 없습니다: {sorted(FILES - found)}")
    print(f"주요 CSV {len(found)}개를 data/raw에 저장했습니다.")


if __name__ == "__main__":
    main()
