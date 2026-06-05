import warnings
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey, linear_reset
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. PARAMÈTRES
# ============================================================

COUNTRY = "MAR"
START_YEAR = 1991
END_YEAR = 2024

OUTPUT_DIR = Path("okun_morocco_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

INDICATORS = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",     # GDP growth annual %
    "unemployment": "SL.UEM.TOTL.ZS"       # Unemployment total % of labor force
}


# ============================================================
# 2. RÉCUPÉRATION WORLD BANK
# ============================================================

def fetch_worldbank_indicator(country, indicator_code, start_year, end_year):
    url = (
        f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator_code}"
        f"?format=json&per_page=20000&date={start_year}:{end_year}"
    )

    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        raise Exception(f"Erreur API World Bank : {response.status_code}")

    data = response.json()

    if len(data) < 2 or data[1] is None:
        return pd.DataFrame(columns=["year", "value"])

    rows = []

    for item in data[1]:
        rows.append({
            "year": int(item["date"]),
            "value": item["value"]
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("year")

    return df


def build_dataset():
    all_series = []

    for variable_name, indicator_code in INDICATORS.items():
        print(f"Téléchargement : {variable_name} | {indicator_code}")

        temp = fetch_worldbank_indicator(
            country=COUNTRY,
            indicator_code=indicator_code,
            start_year=START_YEAR,
            end_year=END_YEAR
        )

        temp = temp.rename(columns={"value": variable_name})
        all_series.append(temp[["year", variable_name]])

    if not all_series:
        return pd.DataFrame()

    df = all_series[0]

    for series in all_series[1:]:
        df = df.merge(series, on="year", how="outer")

    df = df.sort_values("year")
    df = df.set_index("year")

    df.to_csv(OUTPUT_DIR / "raw_worldbank_morocco.csv")

    return df


# ============================================================
# 3. PRÉPARATION DE LA LOI D’OKUN
# ============================================================

def prepare_okun_data(df):
    okun_df = df.copy()

    okun_df["delta_unemployment"] = okun_df["unemployment"].diff()

    okun_df = okun_df.dropna()

    okun_df.to_csv(OUTPUT_DIR / "okun_clean_dataset_morocco.csv")

    return okun_df


# ============================================================
# 4. RAPPORT QUALITÉ DES DONNÉES
# ============================================================

def data_quality_report(df):
    report = pd.DataFrame({
        "variable": df.columns,
        "non_missing_values": df.notna().sum().values,
        "missing_values": df.isna().sum().values,
        "first_available_year": [
            df[col].dropna().index.min() if df[col].notna().sum() > 0 else None
            for col in df.columns
        ],
        "last_available_year": [
            df[col].dropna().index.max() if df[col].notna().sum() > 0 else None
            for col in df.columns
        ]
    })

    report.to_csv(OUTPUT_DIR / "data_quality_report_morocco.csv", index=False)

    print("\n==============================")
    print("RAPPORT QUALITÉ DES DONNÉES - MAROC")
    print("==============================")
    print(report)

    return report


# ============================================================
# 5. TEST ADF DE STATIONNARITÉ
# ============================================================

def adf_test(series, variable_name):
    series = series.dropna()

    if len(series) < 10:
        return {
            "variable": variable_name,
            "adf_p_value": np.nan,
            "conclusion": "Pas assez d'observations"
        }

    result = adfuller(series)

    p_value = result[1]

    if p_value < 0.05:
        conclusion = "Stationnaire"
    else:
        conclusion = "Non stationnaire"

    return {
        "variable": variable_name,
        "adf_p_value": round(p_value, 4),
        "conclusion": conclusion
    }


def run_stationarity_tests(okun_df):
    variables = ["gdp_growth", "unemployment", "delta_unemployment"]

    results = []

    for var in variables:
        results.append(adf_test(okun_df[var], var))

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "stationarity_adf_results_morocco.csv", index=False)

    print("\n==============================")
    print("TESTS ADF - MAROC")
    print("==============================")
    print(results_df)

    return results_df


# ============================================================
# 6. MODÈLE DE BASE DE LA LOI D'OKUN
# ============================================================

def estimate_okun_basic(okun_df):
    y = okun_df["delta_unemployment"]
    X = okun_df[["gdp_growth"]]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    robust_model = model.get_robustcov_results(cov_type="HAC", maxlags=1)

    with open(OUTPUT_DIR / "okun_basic_ols_summary_morocco.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    with open(OUTPUT_DIR / "okun_basic_robust_HAC_summary_morocco.txt", "w", encoding="utf-8") as f:
        f.write(robust_model.summary().as_text())

    print("\n==============================")
    print("LOI D'OKUN - MODÈLE DE BASE - MAROC")
    print("==============================")
    print(model.summary())

    return model, robust_model


# ============================================================
# 7. MODÈLE AVEC RETARDS
# ============================================================

def estimate_okun_with_lags(okun_df):
    lag_df = okun_df.copy()

    lag_df["gdp_growth_lag1"] = lag_df["gdp_growth"].shift(1)
    lag_df["delta_unemployment_lag1"] = lag_df["delta_unemployment"].shift(1)

    lag_df = lag_df.dropna()

    y = lag_df["delta_unemployment"]

    X = lag_df[[
        "gdp_growth",
        "gdp_growth_lag1",
        "delta_unemployment_lag1"
    ]]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    robust_model = model.get_robustcov_results(cov_type="HAC", maxlags=1)

    with open(OUTPUT_DIR / "okun_dynamic_ols_summary_morocco.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    with open(OUTPUT_DIR / "okun_dynamic_robust_HAC_summary_morocco.txt", "w", encoding="utf-8") as f:
        f.write(robust_model.summary().as_text())

    print("\n==============================")
    print("LOI D'OKUN - MODÈLE DYNAMIQUE AVEC RETARDS - MAROC")
    print("==============================")
    print(model.summary())

    return model, robust_model, lag_df


# ============================================================
# 8. TESTS ÉCONOMÉTRIQUES
# ============================================================

def econometric_diagnostics(model, X):
    residuals = model.resid

    dw = durbin_watson(residuals)

    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)

    bp_test = het_breuschpagan(residuals, X)
    bp_stat = bp_test[0]
    bp_pvalue = bp_test[1]

    bg_test = acorr_breusch_godfrey(model, nlags=1)
    bg_stat = bg_test[0]
    bg_pvalue = bg_test[1]

    try:
        reset_test = linear_reset(model, power=2, use_f=True)
        reset_pvalue = reset_test.pvalue
    except Exception:
        reset_pvalue = np.nan

    diagnostics = pd.DataFrame({
        "test": [
            "Durbin-Watson",
            "Jarque-Bera normality",
            "Breusch-Pagan heteroskedasticity",
            "Breusch-Godfrey autocorrelation",
            "Ramsey RESET specification"
        ],
        "statistic": [
            dw,
            jb_stat,
            bp_stat,
            bg_stat,
            np.nan
        ],
        "p_value": [
            np.nan,
            jb_pvalue,
            bp_pvalue,
            bg_pvalue,
            reset_pvalue
        ],
        "interpretation": [
            "Proche de 2 = pas d'autocorrélation forte",
            "p-value > 0.05 = résidus proches d'une distribution normale",
            "p-value > 0.05 = pas d'hétéroscédasticité détectée",
            "p-value > 0.05 = pas d'autocorrélation détectée",
            "p-value > 0.05 = pas de problème majeur de spécification"
        ]
    })

    diagnostics.to_csv(OUTPUT_DIR / "econometric_diagnostics_morocco.csv", index=False)

    print("\n==============================")
    print("TESTS ÉCONOMÉTRIQUES - MAROC")
    print("==============================")
    print(diagnostics)

    return diagnostics


# ============================================================
# 9. TRAIN/TEST POUR ÉVITER LE BIAIS TEMPOREL
# ============================================================

def train_test_okun(okun_df, test_years=5):
    model_df = okun_df[["delta_unemployment", "gdp_growth"]].dropna()

    train = model_df.iloc[:-test_years]
    test = model_df.iloc[-test_years:]

    y_train = train["delta_unemployment"]
    X_train = sm.add_constant(train[["gdp_growth"]])

    y_test = test["delta_unemployment"]
    X_test = sm.add_constant(test[["gdp_growth"]])

    model = sm.OLS(y_train, X_train).fit()

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    prediction_df = pd.DataFrame({
        "actual_delta_unemployment": y_test,
        "predicted_delta_unemployment": predictions
    })

    evaluation = pd.DataFrame({
        "metric": ["MAE", "RMSE"],
        "value": [mae, rmse]
    })

    prediction_df.to_csv(OUTPUT_DIR / "okun_train_test_predictions_morocco.csv")
    evaluation.to_csv(OUTPUT_DIR / "okun_train_test_evaluation_morocco.csv", index=False)

    print("\n==============================")
    print("ÉVALUATION TRAIN/TEST - MAROC")
    print("==============================")
    print(evaluation)

    return model, prediction_df, evaluation


# ============================================================
# 10. GRAPHIQUES
# ============================================================

def plot_okun_scatter(okun_df, model):
    plt.figure(figsize=(8, 6))

    x = okun_df["gdp_growth"]
    y = okun_df["delta_unemployment"]

    plt.scatter(x, y)

    x_line = np.linspace(x.min(), x.max(), 100)
    X_line = sm.add_constant(pd.DataFrame({"gdp_growth": x_line}))
    y_line = model.predict(X_line)

    plt.plot(x_line, y_line)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle="--", linewidth=1)

    plt.title("Loi d'Okun - Maroc")
    plt.xlabel("Croissance du PIB réel annuel (%)")
    plt.ylabel("Variation du taux de chômage")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "okun_scatter_morocco.png")
    plt.show()


def plot_time_series(okun_df):
    plt.figure(figsize=(10, 5))
    plt.plot(okun_df.index, okun_df["gdp_growth"], label="GDP growth")
    plt.plot(okun_df.index, okun_df["delta_unemployment"], label="Delta unemployment")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Maroc - Croissance du PIB et variation du chômage")
    plt.xlabel("Année")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gdp_growth_delta_unemployment_morocco.png")
    plt.show()


def plot_predictions(prediction_df):
    plt.figure(figsize=(10, 5))
    plt.plot(
        prediction_df.index,
        prediction_df["actual_delta_unemployment"],
        label="Réel"
    )
    plt.plot(
        prediction_df.index,
        prediction_df["predicted_delta_unemployment"],
        label="Prédit"
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Prédiction de la variation du chômage - Loi d'Okun - Maroc")
    plt.xlabel("Année")
    plt.ylabel("Variation du chômage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "okun_predictions_morocco.png")
    plt.show()


def plot_residuals(model):
    residuals = model.resid

    plt.figure(figsize=(10, 5))
    plt.plot(residuals.index, residuals)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Résidus du modèle d'Okun - Maroc")
    plt.xlabel("Année")
    plt.ylabel("Résidus")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "okun_residuals_morocco.png")
    plt.show()


def plot_additional_ols_graphs(okun_df, basic_model):
    # Prédictions OLS sur toute la période
    X_ols = sm.add_constant(okun_df[["gdp_growth"]])
    okun_df["predicted_delta_unemployment"] = basic_model.predict(X_ols)

    # 1. Réel vs prédit (toutes années)
    plt.figure(figsize=(10, 5))
    plt.plot(
        okun_df.index,
        okun_df["delta_unemployment"],
        marker="o",
        label="Variation réelle du chômage"
    )
    plt.plot(
        okun_df.index,
        okun_df["predicted_delta_unemployment"],
        marker="o",
        label="Variation prédite par OLS"
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Maroc - Variation du chômage : réel vs prédit OLS")
    plt.xlabel("Année")
    plt.ylabel("Variation annuelle du chômage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ols_real_vs_predicted_okun_morocco.png")
    plt.show()

    # 2. Nuage de points + droite de régression
    plt.figure(figsize=(8, 6))
    plt.scatter(
        okun_df["gdp_growth"],
        okun_df["delta_unemployment"],
        label="Observations"
    )

    x_line = np.linspace(
        okun_df["gdp_growth"].min(),
        okun_df["gdp_growth"].max(),
        100
    )

    alpha = basic_model.params.get("const", 0)
    beta = basic_model.params.get("gdp_growth", 0)
    y_line = alpha + beta * x_line

    plt.plot(
        x_line,
        y_line,
        linewidth=2,
        label="Droite de régression OLS"
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title("Loi d'Okun - Maroc : droite de régression OLS")
    plt.xlabel("Croissance du PIB réel annuel (%)")
    plt.ylabel("Variation annuelle du chômage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ols_regression_line_okun_morocco.png")
    plt.show()

    # 3. Résidus OLS
    plt.figure(figsize=(10, 5))
    plt.plot(
        okun_df.index,
        basic_model.resid,
        marker="o",
        label="Résidus OLS"
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Résidus du modèle OLS - Loi d'Okun - Maroc")
    plt.xlabel("Année")
    plt.ylabel("Résidus")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ols_residuals_okun_morocco.png")
    plt.show()

    # 4. Valeurs observées vs valeurs prédites
    plt.figure(figsize=(8, 6))
    plt.scatter(
        okun_df["delta_unemployment"],
        okun_df["predicted_delta_unemployment"]
    )

    min_val = min(
        okun_df["delta_unemployment"].min(),
        okun_df["predicted_delta_unemployment"].min()
    )

    max_val = max(
        okun_df["delta_unemployment"].max(),
        okun_df["predicted_delta_unemployment"].max()
    )

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=1,
        label="Ligne parfaite"
    )

    plt.title("OLS - Valeurs réelles vs valeurs prédites - Maroc")
    plt.xlabel("Variation réelle du chômage")
    plt.ylabel("Variation prédite du chômage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ols_actual_vs_predicted_scatter_morocco.png")
    plt.show()


# ============================================================
# 11. INTERPRÉTATION AUTOMATIQUE
# ============================================================

def generate_interpretation(model):
    beta = model.params.get("gdp_growth", np.nan)
    pvalue = model.pvalues.get("gdp_growth", np.nan)
    r2 = model.rsquared

    interpretation = []

    interpretation.append("INTERPRÉTATION DE LA LOI D'OKUN - MAROC")
    interpretation.append("")
    interpretation.append("Modèle estimé :")
    interpretation.append("Δu_t = α + β * croissance_PIB_t + ε_t")
    interpretation.append("")
    interpretation.append(f"Coefficient beta estimé : {beta:.4f}")
    interpretation.append(f"P-value du coefficient : {pvalue:.4f}")
    interpretation.append(f"R² du modèle : {r2:.4f}")
    interpretation.append("")

    if beta < 0:
        interpretation.append(
            "Le coefficient est négatif. Cela signifie qu'une hausse de la croissance du PIB est associée à une baisse de la variation du chômage."
        )
    else:
        interpretation.append(
            "Le coefficient est positif. Le résultat ne correspond pas à la relation théorique attendue par la loi d'Okun."
        )

    if pvalue < 0.05:
        interpretation.append(
            "Le coefficient est statistiquement significatif au seuil de 5%."
        )
    else:
        interpretation.append(
            "Le coefficient n'est pas statistiquement significatif au seuil de 5%."
        )

    interpretation.append("")
    interpretation.append(
        "Attention : ce modèle montre une relation statistique, mais il ne suffit pas seul pour prouver une causalité économique."
    )

    text = "\n".join(interpretation)

    with open(OUTPUT_DIR / "interpretation_okun_morocco.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("\n==============================")
    print(text)
    print("==============================")

    return text


# ============================================================
# 12. PIPELINE FINAL
# ============================================================

def main():
    print("\nRécupération des données World Bank pour le Maroc...\n")

    raw_df = build_dataset()

    if raw_df.empty:
        print("Aucune donnée récupérée. Vérifie le code pays ou la connectivité.")
        return

    data_quality_report(raw_df)

    okun_df = prepare_okun_data(raw_df)

    run_stationarity_tests(okun_df)

    basic_model, basic_robust_model = estimate_okun_basic(okun_df)

    dynamic_model, dynamic_robust_model, lag_df = estimate_okun_with_lags(okun_df)

    X_basic = sm.add_constant(okun_df[["gdp_growth"]])
    econometric_diagnostics(basic_model, X_basic)

    test_model, prediction_df, evaluation = train_test_okun(okun_df, test_years=5)

    plot_okun_scatter(okun_df, basic_model)
    plot_time_series(okun_df)
    plot_predictions(prediction_df)
    plot_residuals(basic_model)

    generate_interpretation(basic_model)

    print("\nTerminé.")
    print(f"Tous les fichiers sont dans le dossier : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
