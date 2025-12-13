import numpy as np
import pandas as pd

'''
1900 - 3%
Economiile europene la început de secol prezentau o inflație redusă și relativ stabilă.
Pentru Regatul României, literatura economică arată creșteri generale de prețuri între 2–4%, deci 3% este un punct realist de start.

1914 - 4%
Înainte de Primul Război Mondial, economia rămâne stabilă, dar:
-crește cheltuiala militară,
-se intensifică importurile și presiunea pe valută.
Acest lucru justifică o inflație moderată ușor peste nivelul de la 1900.

1918 - 20%

Finalul Primului Război Mondial aduce:
-penurie de resurse,
-deprecierea masivă a monedei,
-dezechilibre bugetare și inflație accelerată.
Estimările istorice arată inflație anuală între 15–30% în țările afectate.
Valoarea 20% este o ancoră conservatoare și realistă.

1938 - 3.5%
Perioada interbelică târzie a fost relativ stabilă în Europa de Est:
-producție agricolă solidă,
-reforme monetare,
-echilibru macroeconomic.
O inflație de 3–4% este consistentă cu sursele istorice.

1945 - 25%
Sfârșitul celui de-al Doilea Război Mondial a produs:
-hiperinflație în unele țări,
-pierderi industriale,
-dezechilibre majore pe piața alimentară.
România, fiind în tabăra învinsă și cu schimbări politice dramatice, a avut inflație ridicată.
25% este o valoare realistă și în zona raportată în surse economice.

1950 - 2.5%
În perioada 1948–1950 are loc stabilizarea forțată a prețurilor sub regimul comunist:
-control strict asupra prețurilor,
-statul fixează prețurile bunurilor de bază,
-eliminarea mecanismelor de piață.
Inflația scade artificial în zona 2–3%, uneori chiar mai jos.

1970 - 3%
Anii ’60–’70 în România sunt caracterizați de:
-industrializare accelerată,
-control strict al prețurilor,
-variații mici în indicele prețurilor de consum.
În majoritatea surselor, inflația era între 2–4%, deci 3% este un anchor adecvat.
'''

anchors = {
    1900: 3.0,
    1914: 4.0,
    1918: 20.0,
    1938: 3.5,
    1945: 25.0,
    1950: 2.5,
    1970: 3.0,
}

def smooth_interpolate(anchors):
    years = np.arange(min(anchors), max(anchors) + 1)
    anchor_years = np.array(list(anchors.keys()))
    anchor_vals = np.array(list(anchors.values()))

    #interpolare logaritmica (smoother)
    log_vals = np.log(anchor_vals)
    interp = np.interp(years, anchor_years, log_vals)

    return pd.DataFrame({
        "An": years,
        "rata_infl": np.exp(interp).round(2)
    })

df_hist = smooth_interpolate(anchors)
df_hist.to_csv("inflation_1900_1970_model.csv", index=False)