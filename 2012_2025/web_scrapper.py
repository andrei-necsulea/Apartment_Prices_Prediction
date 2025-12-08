import re
import ast
from datetime import date
import requests
import pandas as pd

URL = "https://www.imobiliare.ro/indicele-imobiliare-ro/craiova"

def download_html(url: str) -> str:
    """descarca HTML-ul paginii si il returneaza ca text."""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) "
            "Gecko/20100101 Firefox/115.0"
        )
    }

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    html = resp.text

    with open("craiova_index_page.html", "w", encoding="utf-8") as f:
        f.write(html)

    return html


def extract_chart_block(html: str) -> str:
    """
    caut blocul de cod JS care defineste graficul nostru (chartIndice_6)
    si returneaza doar bucata aceea (ca text).
    """

    marker = "var graficIndice = new Chart(chartIndice_6"
    start_idx = html.find(marker)

    if start_idx == -1:
        raise RuntimeError("Nu am gasit blocul cu indicele!")

    # luam tot pana la 'options:' (restul nu ne intereseaza pentru ca sunt doar proprietati)
    end_idx = html.find("options:", start_idx)

    if end_idx == -1:
        # fallback: luam o bucata fixa daca nu gasim options
        end_idx = start_idx + 8000

    block = html[start_idx:end_idx]

    return block


def extract_labels_and_data(block: str):
    """
    Din blocul de JS extragem:
      labels: ['7/2012', ...]
      data:   [731, 727, ...]
    si le convertim in liste Python.
    """

    #labels:
    labels_match = re.search(r"labels\s*:\s*\[(.*?)\]", block, re.S)

    if not labels_match:
        raise RuntimeError("Nu am gasit labels[...] in blocul de chart!")

    labels_str = "[" + labels_match.group(1) + "]"
    #pentru transformarea unei liste string in lista python : "[...]" la [..]
    labels = ast.literal_eval(labels_str)

    #data:
    data_match = re.search(r"data\s*:\s*\[(.*?)\]", block, re.S)

    if not data_match:
        raise RuntimeError("Nu am gasit data[...] in blocul de chart!")

    data_str = "[" + data_match.group(1) + "]"
    values = ast.literal_eval(data_str)

    if len(labels) != len(values):
        raise RuntimeError(
            f"Lungime diferita intre labels ({len(labels)}) si valori ({len(values)})!"
        )

    return labels, values


def build_dataframe(labels, values) -> pd.DataFrame:
    """
    Construieste un DataFrame cu coloanele:
      - date (prima zi din luna)
      - price_per_sqm
    """

    rows = []
    for label, val in zip(labels, values):
        # label este de forma '7/2012' (luna/an)
        month_str, year_str = label.split("/")
        year = int(year_str)
        month = int(month_str)

        d = date(year, month, 1)
        rows.append({"date": d, "price_per_sqm": float(val)})

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)

    return df


def main():
    html = download_html(URL)

    block = extract_chart_block(html)

    labels, values = extract_labels_and_data(block)

    df = build_dataframe(labels, values)

    df.to_csv("craiova_apartment_prices_2012_2025_euro.csv", index=False)

    print("\nAm salvat fisierul cu succes!\n")
    print("craiova_apartment_prices_2012_2025_euro.csv")

if __name__ == "__main__":
    main()