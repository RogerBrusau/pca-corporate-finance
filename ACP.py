import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Variables objectiu de l'ACP segons la definicio operativa
PCA_VARS_TARGET = [
    "log_market_cap",
    "ROE",
    "margen_operativo",
    "rotacion_activos",
    "deuda_neta_sobre_EBITDA",
    "EV_EBITDA",
    "crecimiento_ventas_1y",
]

META_COLS_CANDIDATAS = ["ticker", "company", "sector", "market_cap_usd"]

def parse_args():
    ap = argparse.ArgumentParser(description="ACP Top50 S&P500 des de CSV local.")
    ap.add_argument("--csv", required=True, help="Ruta al CSV amb dades.")
    ap.add_argument("--umbral", type=float, default=0.9,
                    help="Llindar de variancia acumulada per triar d.")
    ap.add_argument("--winsor", type=float, default=0.0,
                    help="Winsoritzacio bilateral, per exemple 0.01. 0 desactiva.")
    ap.add_argument("--max_na_frac", type=float, default=0.2,
                    help="Fraccio maxima de NaN permesa per variable abans d'excloure-la.")
    ap.add_argument("--outdir", default="outputs", help="Carpeta de sortida.")
    ap.add_argument("--label_col", default="ticker",
                help="Columna per a etiquetes: ticker o company. none per no etiquetar.")
    ap.add_argument("--label_k", type=int, default=25,
                help="Nombre maxim de punts a etiquetar, prioritzant els més allunyats del centre.")

    return ap.parse_args()

def winsorize(df, p):
    if p <= 0:
        return df
    lower = df.quantile(p)
    upper = df.quantile(1 - p)
    return df.clip(lower=lower, upper=upper, axis="columns")

def triar_d(var_acum, umbral):
    return int(np.searchsorted(var_acum, umbral) + 1)

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    (outdir / "figs").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    # 1) Llegir el CSV
    df = pd.read_csv(args.csv)

    # 2) Determinar les variables disponibles per a l'ACP
    disponibles = [v for v in PCA_VARS_TARGET if v in df.columns]
    faltantes = [v for v in PCA_VARS_TARGET if v not in df.columns]
    if len(disponibles) < 2:
        raise ValueError(
            f"No hi ha prou variables de l'ACP al CSV. Falten: {faltantes}"
        )

    meta_cols = [c for c in META_COLS_CANDIDATAS if c in df.columns]
    X_raw = df[disponibles].copy()

    # 3) Filtrar pel percentatge de NaN i imputar amb la mediana
    frac_nan = X_raw.isna().mean()
    keep_vars = frac_nan[frac_nan <= args.max_na_frac].index.tolist()
    dropped_by_nan = [v for v in disponibles if v not in keep_vars]
    if len(keep_vars) < 2:
        raise ValueError(
            "Després de filtrar per NaN queden menys de dues variables per a l'ACP."
        )
    X = X_raw[keep_vars].copy()
    X = X.fillna(X.median(numeric_only=True))

    # 4) Winsoritzacio opcional i estandarditzacio
    Xw = winsorize(X, args.winsor) if args.winsor > 0 else X
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(Xw.values)
    Z = np.asarray(Z)

    # 5) ACP
    pca = PCA(svd_solver="full")
    T = pca.fit_transform(Z)                 # scores
    eigvals = pca.explained_variance_        # valors propis
    var_exp = pca.explained_variance_ratio_  # fraccio per component
    var_acum = np.cumsum(var_exp)
    d = triar_d(var_acum, args.umbral)
    loadings = pca.components_.T             # columnes = components

    # 6) Desar taules
    ev_df = pd.DataFrame({
        "component": np.arange(1, len(eigvals) + 1),
        "eigenvalue": eigvals,
        "explained_ratio": var_exp,
        "cumulative_ratio": var_acum
    })
    ev_df.to_csv(outdir / "tables" / "eigenvalues.csv", index=False)

    load_df = pd.DataFrame(
        loadings,
        index=keep_vars,
        columns=[f"PC{i}" for i in range(1, loadings.shape[1] + 1)]
    )
    load_df.to_csv(outdir / "tables" / "loadings.csv")

    scores_df = pd.DataFrame(
        T, columns=[f"PC{i}" for i in range(1, T.shape[1] + 1)]
    )
    if meta_cols:
        scores_df = pd.concat([df[meta_cols].reset_index(drop=True), scores_df], axis=1)
    scores_df.to_csv(outdir / "tables" / "scores.csv", index=False)

    # 7) Grafics
    plt.figure()
    plt.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o")
    plt.xlabel("Component")
    plt.ylabel("Valor propi")
    plt.title("Corba de pedreny")
    plt.tight_layout()
    plt.savefig(outdir / "figs" / "scree.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(np.arange(1, len(var_acum) + 1), var_acum, marker="o")
    plt.axhline(args.umbral, linestyle="--")
    plt.xlabel("Component")
    plt.ylabel("Variancia acumulada")
    plt.title(f"Variancia acumulada i llindar {args.umbral:.2f}")
    plt.tight_layout()
    plt.savefig(outdir / "figs" / "varianza_acumulada.png", dpi=150)
    plt.close()

    if T.shape[1] >= 2:
        plt.figure()
        plt.scatter(T[:, 0], T[:, 1], alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Biplot basic PC1 i PC2")

        # vectors de carregues
        L = loadings[:, :2]
        escala = 3.0
        for i, var in enumerate(keep_vars):
            plt.arrow(0, 0, escala * L[i, 0], escala * L[i, 1],
                      head_width=0.05, length_includes_head=True)
            plt.text(escala * L[i, 0] * 1.05, escala * L[i, 1] * 1.05,
                     var, fontsize=8)

        # etiquetes dels punts
        label_col = args.label_col
        if label_col != "none" and label_col in df.columns:
            labels = df[label_col].astype(str).values
        elif label_col != "none":
            labels = df.index.astype(str).values
        else:
            labels = None

        if labels is not None:
            # prioritza els més allunyats de l'origen en el pla PC1 i PC2
            r = np.sqrt(T[:, 0]**2 + T[:, 1]**2)
            k = max(1, min(args.label_k, len(r)))
            idx = np.argsort(-r)[:k]

            # desplaçament relatiu per reduir solapaments
            dx = 0.015 * (T[:, 0].max() - T[:, 0].min())
            dy = 0.015 * (T[:, 1].max() - T[:, 1].min())

            for i in idx:
                x, y = T[i, 0], T[i, 1]
                offx = dx if x >= 0 else -dx
                offy = dy if y >= 0 else -dy
                plt.annotate(labels[i], (x, y),
                             xytext=(x + offx, y + offy),
                             textcoords="data",
                             fontsize=8, alpha=0.9,
                             arrowprops=dict(arrowstyle="-", lw=0.3, alpha=0.6))

        plt.tight_layout()
        plt.savefig(outdir / "figs" / "biplot_pc1_pc2.png", dpi=150)
        plt.close()

    # 8) Metadades
    meta = {
        "csv": str(Path(args.csv).resolve()),
        "pca_vars_target": PCA_VARS_TARGET,
        "pca_vars_disponibles": disponibles,
        "pca_vars_faltantes_en_csv": faltantes,
        "pca_vars_descartadas_por_nan": dropped_by_nan,
        "pca_vars_utilizadas": keep_vars,
        "winsor": args.winsor,
        "umbral_varianza": args.umbral,
        "d_elegido": int(d),
        "media": dict(zip(keep_vars, scaler.mean_.tolist())),
        "std": dict(zip(keep_vars, scaler.scale_.tolist()))
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Variables objectiu: {PCA_VARS_TARGET}")
    print(f"Utilitzades a l'ACP:  {keep_vars}")
    print(f"Components seleccionats d = {d}")
    print(f"Resultats a {outdir.resolve()}")

if __name__ == "__main__":
    main()
