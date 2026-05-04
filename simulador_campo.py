"""
SIMULADOR DE CAMPO — Sistema ALMMo-0 Irrigação Inteligente
Simula o comportamento completo do loop diário do Raspberry Pi,
substituindo hardware real por dados sintéticos:
  - Arduino (sensor capacitivo) → tensão gerada por perfil simulado
  - OpenWeatherMap API → dados climáticos reais da NASA POWER
  - Solenoide → log de tempo calculado (sem actuação física)
Executa N dias de simulação e gera relatório completo em CSV.
"""
import pickle
import numpy as np
import csv
import os
import json
from datetime import date, timedelta
from collections import Counter

# =============================================================================
# ALMMo-0 v3.0 — necessário para reconstruir modelos a partir do pkl
# =============================================================================
class ALMMo0:
    def __init__(self, n_inputs=5, r_threshold=0.5, max_rules=50,
                 age_limit=100, epsilon=1e-8, n_classes=3,
                 min_rules_per_class=3):
        self.n_inputs=n_inputs; self.r_threshold=r_threshold
        self.max_rules=max_rules; self.age_limit=age_limit
        self.epsilon=epsilon; self.n_classes=n_classes
        self.min_rules_per_class=min_rules_per_class
        self.rules=[]; self.input_mean=np.zeros(n_inputs)
        self.input_std=np.ones(n_inputs)
        self.n_samples_seen=0; self.n_rules_created=0; self.n_rules_pruned=0

    def fit_normalizer(self,X):
        self.input_mean=np.mean(X,axis=0); self.input_std=np.std(X,axis=0)
        self.input_std[self.input_std<self.epsilon]=1.0

    def normalize(self,x): return (x-self.input_mean)/self.input_std

    def _dist(self,a,b):
        d=a-b; return float(np.sqrt(np.dot(d,d)))

    def _dists(self,x_norm):
        if not self.rules: return np.array([])
        return np.array([self._dist(x_norm,r['center']) for r in self.rules])

    def _create(self,x_norm,y):
        self.rules.append({'center':x_norm.copy(),'consequent':int(y),'age':0,'activations':1})
        self.n_rules_created+=1

    def predict(self,x):
        x_n=self.normalize(x); d=self._dists(x_n)
        w=1.0/(d**2+self.epsilon)
        v=np.zeros(self.n_classes)
        for i,r in enumerate(self.rules): v[r['consequent']]+=w[i]
        return int(np.argmax(v))

    def learn(self,x,y):
        self.n_samples_seen+=1; x_n=self.normalize(x)
        if not self.rules: self._create(x_n,y); return
        d=self._dists(x_n); idx=int(np.argmin(d)); dm=d[idx]
        if dm>self.r_threshold: self._create(x_n,y)
        else: self._update(idx,x_n,y,dm)
        self._age(idx); self._prune()

    def _update(self,idx,x_n,y,dist):
        r=self.rules[idx]; r['activations']+=1; eta=1.0/r['activations']
        r['center']=r['center']+eta*(x_n-r['center'])
        if r['consequent']!=y:
            af=max(0.0,1.0-dist/self.r_threshold); ee=eta*af
            if ee>0:
                nc=(1-ee)*r['consequent']+ee*y
                r['consequent']=int(np.round(np.clip(nc,0,self.n_classes-1)))
        r['age']=0

    def _age(self,idx):
        for i,r in enumerate(self.rules):
            if i!=idx: r['age']+=1

    def _prune(self):
        n0=len(self.rules); cc=Counter(r['consequent'] for r in self.rules)
        surv=[]
        for r in self.rules:
            if r['age']<=self.age_limit: surv.append(r)
            else:
                if cc[r['consequent']]>self.min_rules_per_class: cc[r['consequent']]-=1
                else: r['age']=0; surv.append(r)
        if len(surv)>self.max_rules:
            surv.sort(key=lambda r:r['age'],reverse=True)
            cc2=Counter(r['consequent'] for r in surv); kept=[]
            for r in surv:
                if len(kept)>=self.max_rules:
                    if cc2[r['consequent']]<=self.min_rules_per_class: kept.append(r)
                    else: cc2[r['consequent']]-=1
                else: kept.append(r)
            surv=kept
        self.rules=surv; self.n_rules_pruned+=n0-len(self.rules)

    @classmethod
    def from_dict(cls, d):
        m = cls(n_inputs=d['n_inputs'], r_threshold=d['r_threshold'],
                max_rules=d['max_rules'], age_limit=d['age_limit'],
                n_classes=d.get('n_classes',3),
                min_rules_per_class=d.get('min_rules_per_class',3))
        m.rules = d['rules']
        m.input_mean = np.array(d['input_mean'])
        m.input_std  = np.array(d['input_std'])
        m.n_samples_seen  = d.get('n_samples_seen', 0)
        m.n_rules_created = d.get('n_rules_created', 0)
        m.n_rules_pruned  = d.get('n_rules_pruned', 0)
        return m

# =============================================================================
# CONFIGURAÇÃO DO CANTEIRO E SISTEMA
# =============================================================================
CONFIG = {
    # Canteiro experimental
    "area_m2": 0.21 * 0.355,          # 0.0746 m²
    # Solenoide — PLACEHOLDER até ter a vazão real
    "vazao_l_por_min": None,           # ← PREENCHER com vazão real (L/min)
    "vazao_placeholder": 1.0,          # usado apenas quando None
    # Doses alvo por classe (mm) — baseadas nos dados de treino v11
    "dose_c1_mm": 5.0,
    "dose_c2_mm": 40.0,
    # Data de plantio (início da estação chuvosa 2021 — janela agronômica correta para soja no MA)
    "data_plantio": date(2021, 10, 1),
    # Modelo
    "pkl_path": "memoria_cold_start_v12_ensemble.pkl",
    "log_path": "log_campo_simulado.csv",
    "pkl_campo_path": "memoria_ensemble_campo.pkl",
    # Horário de decisão (simulado)
    "hora_decisao": "06:00",
}

# =============================================================================
# MÓDULO 1 — CÁLCULO DE TEMPO DE SOLENOIDE
# =============================================================================
def calcular_tempo_solenoide(classe: int, config: dict) -> dict:
    area = config["area_m2"]
    vazao = config["vazao_l_por_min"] or config["vazao_placeholder"]
    placeholder = config["vazao_l_por_min"] is None
    if classe == 0:
        return {
            "tempo_segundos": 0,
            "volume_ml": 0,
            "dose_mm": 0.0,
            "observacao": "Sem irrigação"
        }
    dose_mm = config["dose_c1_mm"] if classe == 1 else config["dose_c2_mm"]
    volume_l = dose_mm * area
    volume_ml = volume_l * 1000
    tempo_min = volume_l / vazao
    tempo_seg = round(tempo_min * 60)
    return {
        "tempo_segundos": tempo_seg,
        "volume_ml": round(volume_ml, 1),
        "dose_mm": dose_mm,
        "observacao": f"{'[PLACEHOLDER vazão=' + str(vazao) + ' L/min] ' if placeholder else ''}Classe C{classe}"
    }

# =============================================================================
# MÓDULO 2 — SIMULADOR DE SENSOR (substitui Arduino)
# =============================================================================
class SimuladorArduino:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.tensao_anterior = 20.0
        self.estado = "inicial"

    def ler_tensao(self, dap: int, irrigou_ontem: bool, chuva_ontem_mm: float) -> float:
        if dap < 30:
            taxa_base = self.rng.uniform(1.5, 3.0)
        elif dap < 60:
            taxa_base = self.rng.uniform(2.5, 4.5)
        elif dap < 90:
            taxa_base = self.rng.uniform(3.0, 6.0)
        else:
            taxa_base = self.rng.uniform(1.5, 3.0)
        if irrigou_ontem or chuva_ontem_mm > 5.0:
            reducao = self.rng.uniform(8.0, 20.0)
            nova_tensao = max(5.0, self.tensao_anterior - reducao + taxa_base)
        else:
            ruido = self.rng.normal(0, 0.8)
            nova_tensao = self.tensao_anterior + taxa_base + ruido
        nova_tensao = float(np.clip(nova_tensao, 4.0, 67.0)) 
        self.tensao_anterior = nova_tensao
        return round(nova_tensao, 2)

# =============================================================================
# MÓDULO 3 — NASA POWER API (substitui SimuladorOpenWeatherMap sintético)
# =============================================================================
import requests

class SimuladorOpenWeatherMap:
    """
    Dados reais da NASA POWER para Imperatriz-MA.
    API: NASA POWER Daily Point (sem API key necessária)
    Localização: lat=-5.5256, lon=-47.4687
    """

    LAT = -5.5256
    LON = -47.4687
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    def __init__(self, seed=None):  # seed ignorado, mantido por compatibilidade
        self.historico_chuva = []
        self.historico_tmax  = []
        self._cache = {}

    def _buscar_nasa_power(self, data: date) -> dict:
        """Busca dados diários da NASA POWER para Imperatriz-MA."""
        chave = data.isoformat()
        if chave in self._cache:
            return self._cache[chave]

        data_str = data.strftime("%Y%m%d")
        params = {
            "parameters": "PRECTOTCORR,T2M_MAX",
            "community": "AG",
            "longitude": self.LON,
            "latitude": self.LAT,
            "start": data_str,
            "end": data_str,
            "format": "JSON",
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            dados = resp.json()["properties"]["parameter"]

            chuva = dados["PRECTOTCORR"].get(data_str, 0.0)
            tmax  = dados["T2M_MAX"].get(data_str, 31.0)

            if chuva < 0:
                chuva = 0.0
            if tmax < -100:
                tmax = 31.0

            resultado = {"chuva_mm": round(float(chuva), 2),
                         "tmax_c":   round(float(tmax), 2)}
            self._cache[chave] = resultado
            return resultado

        except Exception as e:
            print(f"  [NASA POWER] Erro para {data}: {e} — usando fallback")
            mes = data.month
            fallback_chuva = {1:12,2:10,3:9,4:5,5:2,6:0,
                               7:0,8:0,9:1,10:4,11:8,12:11}
            fallback_tmax  = {1:31,2:31,3:32,4:33,5:34,6:34,
                               7:35,8:36,9:36,10:34,11:32,12:31}
            return {"chuva_mm": float(fallback_chuva.get(mes, 5)),
                    "tmax_c":   float(fallback_tmax.get(mes, 32))}

    def obter_dados(self, data: date) -> dict:
        """
        Busca os 3 dias anteriores à data para calcular acumulados.
        Interface mantida igual ao simulador original.
        """
        chuvas = []
        tmaxes = []

        for delta in range(2, -1, -1):   # D-2, D-1, D
            d = data - timedelta(days=delta)
            resultado = self._buscar_nasa_power(d)
            chuvas.append(resultado["chuva_mm"])
            tmaxes.append(resultado["tmax_c"])

        chuva_hoje = chuvas[-1]

        return {
            "chuva_acum_3d_mm": round(sum(chuvas), 2),
            "tmax_max_3d_c":    round(max(tmaxes), 2),
            "chuva_hoje_mm":    chuva_hoje,
        }

# =============================================================================
# MÓDULO 4 — FEATURE ENGINEERING
# =============================================================================
def calcular_features(
    tensao_hoje: float,
    tensao_ontem: float,
    chuva_acum_3d: float,
    tmax_max_3d: float,
    dap: int
) -> np.ndarray:
    delta_tensao = tensao_hoje - tensao_ontem
    features = np.array([
        tensao_hoje,
        chuva_acum_3d,
        tmax_max_3d,
        float(dap),
        delta_tensao
    ], dtype=np.float64)
    return features

# =============================================================================
# MÓDULO 5 — ENSEMBLE: INFERÊNCIA + APRENDIZADO ONLINE
# =============================================================================
class EnsembleALMMo0:
    def __init__(self, pkl_path: str):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Modelo não encontrado: '{pkl_path}'\n"
                f"Certifica-te que o ficheiro pkl do cold start v12 está "
                f"na mesma pasta que este script."
            )
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)

        # pkl do cold_start_v12 guarda dict com chave "models" (lista de dicts)
        if isinstance(raw, dict) and "models" in raw:
            lista_dicts = raw["models"]
            self.modelos = [ALMMo0.from_dict(d) for d in lista_dicts]
        elif isinstance(raw, dict) and "modelos" in raw:
            lista_dicts = raw["modelos"]
            self.modelos = [ALMMo0.from_dict(d) if isinstance(d, dict) else d
                            for d in lista_dicts]
        elif isinstance(raw, list):
            self.modelos = [ALMMo0.from_dict(d) if isinstance(d, dict) else d
                            for d in raw]
        else:
            raise ValueError(f"Formato do pkl não reconhecido: {type(raw)}")

        print(f"[Ensemble] Carregado: {len(self.modelos)} modelos")

    # Pesos por classe: reflectem o custo assimétrico de errar em irrigação.
    # Não irrigar quando devia (stress da planta) custa mais do que
    # irrigar quando não precisava (desperdício de água).
    PESO_CLASSE = {0: 1.0, 1: 1.5, 2: 2.0}

    def predizer(self, x: np.ndarray) -> dict:
        votos_brutos = Counter(modelo.predict(x) for modelo in self.modelos)
        # Votação ponderada pelo custo assimétrico
        score = {c: votos_brutos.get(c, 0) * self.PESO_CLASSE[c] for c in range(3)}
        classe_final = max(score, key=score.get)
        consenso = votos_brutos.get(classe_final, 0) / len(self.modelos)
        return {
            "classe": classe_final,
            "votos": dict(votos_brutos),
            "consenso_pct": round(consenso * 100, 1),
            "n_modelos": len(self.modelos)
        }

    def aprender(self, x: np.ndarray, y: int):
        for modelo in self.modelos:
            modelo.learn(x, y)

    def salvar(self, pkl_path: str):
        with open(pkl_path, "wb") as f:
            pickle.dump(self.modelos, f)

    def estatisticas_regras(self) -> dict:
        total = {0: 0, 1: 0, 2: 0}
        for modelo in self.modelos:
            for rule in modelo.rules:
                c = rule["consequent"]
                total[c] = total.get(c, 0) + 1
        total["total"] = sum(total.values())
        return total

# =============================================================================
# MÓDULO 6 — LOGGER
# =============================================================================
class Logger:
    CABECALHO = [
        "data", "dap", "tensao_solo_kpa", "tensao_ontem_kpa",
        "delta_tensao_kpa", "chuva_acum_3d_mm", "tmax_max_3d_c",
        "classe_predita", "consenso_pct", "votos_c0", "votos_c1", "votos_c2",
        "dose_mm", "volume_ml", "tempo_solenoide_seg",
        "n_regras_total", "n_regras_c0", "n_regras_c1", "n_regras_c2",
        "observacao"
    ]

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._inicializar()

    def _inicializar(self):
        # Sempre recria o CSV no início de cada simulação (evita acumulação entre execuções)
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.CABECALHO)

    def registar(self, data: date, dap: int, features: np.ndarray,
                 tensao_ontem: float, resultado_pred: dict,
                 resultado_dose: dict, stats_regras: dict):
        linha = [
            data.isoformat(),
            dap,
            round(features[0], 2),
            round(tensao_ontem, 2),
            round(features[4], 2),
            round(features[1], 2),
            round(features[2], 2),
            resultado_pred["classe"],
            resultado_pred["consenso_pct"],
            resultado_pred["votos"].get(0, 0),
            resultado_pred["votos"].get(1, 0),
            resultado_pred["votos"].get(2, 0),
            resultado_dose["dose_mm"],
            resultado_dose["volume_ml"],
            resultado_dose["tempo_segundos"],
            stats_regras["total"],
            stats_regras.get(0, 0),
            stats_regras.get(1, 0),
            stats_regras.get(2, 0),
            resultado_dose["observacao"],
        ]
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(linha)

# =============================================================================
# LOOP DIÁRIO PRINCIPAL
# =============================================================================
def loop_diario(data: date, estado: dict, ensemble: EnsembleALMMo0,
                arduino: SimuladorArduino, meteo: SimuladorOpenWeatherMap,
                logger: Logger, config: dict, verbose: bool = True) -> dict:
    tensao_hoje = arduino.ler_tensao(
        dap=estado["dap"],
        irrigou_ontem=estado["irrigou_ontem"],
        chuva_ontem_mm=estado["chuva_ontem_mm"]
    )
    dados_meteo = meteo.obter_dados(data)
    features = calcular_features(
        tensao_hoje=tensao_hoje,
        tensao_ontem=estado["tensao_ontem"],
        chuva_acum_3d=dados_meteo["chuva_acum_3d_mm"],
        tmax_max_3d=dados_meteo["tmax_max_3d_c"],
        dap=estado["dap"]
    )
    resultado_pred = ensemble.predizer(features)
    classe = resultado_pred["classe"]
    resultado_dose = calcular_tempo_solenoide(classe, config)
    ensemble.aprender(features, classe)
    ensemble.salvar(config["pkl_campo_path"])
    stats = ensemble.estatisticas_regras()
    logger.registar(data, estado["dap"], features, estado["tensao_ontem"],
                    resultado_pred, resultado_dose, stats)
    if verbose:
        nomes = {0: "C0-SemIrrig", 1: "C1-Manutencao", 2: "C2-Intensiva"}
        print(f"  {data} | DAP={estado['dap']:3d} | "
              f"Tensão={tensao_hoje:.1f}kPa | Δ={features[4]:+.1f} | "
              f"Chuva={dados_meteo['chuva_acum_3d_mm']:.1f}mm | "
              f"Tmax={dados_meteo['tmax_max_3d_c']:.1f}°C | "
              f"→ {nomes[classe]} [{resultado_pred['consenso_pct']:.0f}% consenso] | "
              f"Dose={resultado_dose['dose_mm']}mm | "
              f"Solenoide={resultado_dose['tempo_segundos']}s | "
              f"Regras={stats['total']}")
    novo_estado = {
        "dap": estado["dap"] + 1,
        "tensao_ontem": tensao_hoje,
        "irrigou_ontem": classe > 0,
        "chuva_ontem_mm": dados_meteo["chuva_hoje_mm"],
    }
    return novo_estado

# =============================================================================
# RELATÓRIO FINAL
# =============================================================================
def gerar_relatorio(log_path: str, n_dias: int, config: dict):
    import csv
    linhas = []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        linhas = list(reader)
    if not linhas:
        print("Log vazio.")
        return
    classes = [int(l["classe_predita"]) for l in linhas]
    tempos  = [int(l["tempo_solenoide_seg"]) for l in linhas]
    tensoes = [float(l["tensao_solo_kpa"]) for l in linhas]
    regras  = [int(l["n_regras_total"]) for l in linhas]
    contagem = Counter(classes)
    total_irrig = sum(t for t in tempos if t > 0)
    vazao = config["vazao_l_por_min"] or config["vazao_placeholder"]
    volume_total_l = (total_irrig / 60) * vazao
    print("\n" + "="*65)
    print("  RELATÓRIO DA SIMULAÇÃO DE CAMPO")
    print("="*65)
    print(f"  Dias simulados      : {n_dias}")
    print(f"  Período             : {linhas[0]['data']} → {linhas[-1]['data']}")
    print(f"  Canteiro            : {config['area_m2']*10000:.1f} cm² "
          f"({config['area_m2']:.4f} m²)")
    print(f"  Vazão solenoide     : {vazao} L/min "
          f"{'[PLACEHOLDER]' if config['vazao_l_por_min'] is None else ''}")
    print("-"*65)
    print(f"  Dias sem irrigação  : {contagem[0]} ({100*contagem[0]/n_dias:.1f}%)")
    print(f"  Dias manutenção C1  : {contagem[1]} ({100*contagem[1]/n_dias:.1f}%)")
    print(f"  Dias intensiva C2   : {contagem[2]} ({100*contagem[2]/n_dias:.1f}%)")
    print("-"*65)
    print(f"  Total solenoide     : {total_irrig}s "
          f"({total_irrig/60:.1f} min)")
    print(f"  Volume total aprox  : {volume_total_l:.2f} L "
          f"[com vazão {vazao} L/min]")
    print(f"  Tensão média        : {np.mean(tensoes):.1f} kPa")
    print(f"  Tensão min/max      : {min(tensoes):.1f} / {max(tensoes):.1f} kPa")
    print(f"  Regras no final     : {regras[-1]} (início: {regras[0]})")
    print("="*65)
    print(f"  Log completo salvo em: {log_path}")
    print("="*65 + "\n")

# =============================================================================
# PONTO DE ENTRADA
# =============================================================================
def main():
    print("="*65)
    print("  SIMULADOR DE CAMPO — ALMMo-0 Irrigação Inteligente")
    print("  IFMA Imperatriz | TCC Gabriel Barros | 2026")
    print("="*65)
    N_DIAS = 94
    DAP_INICIAL = 14
    DATA_INICIO = CONFIG["data_plantio"] + timedelta(days=DAP_INICIAL)
    if not os.path.exists(CONFIG["pkl_path"]):
        print(f"\n[AVISO] Ficheiro '{CONFIG['pkl_path']}' não encontrado.")
        print("        A executar em modo DEMO sem modelo real.")
        MODO_DEMO = True
    else:
        MODO_DEMO = False
    if not MODO_DEMO:
        ensemble = EnsembleALMMo0(CONFIG["pkl_path"])
    else:
        class EnsembleDemo:
            def predizer(self, x):
                tensao = x[0]; delta = x[4]
                if tensao < 25: c = 0
                elif tensao < 38 and abs(delta) < 1: c = 1
                elif delta > 3: c = 2
                else: c = 0
                return {"classe": c, "votos": {c: 14},
                        "consenso_pct": 100.0, "n_modelos": 14}
            def aprender(self, x, y): pass
            def salvar(self, path): pass
            def estatisticas_regras(self):
                return {0: 10, 1: 5, 2: 3, "total": 18}
        ensemble = EnsembleDemo()
        print("[DEMO] Usando preditor simplificado.")

    arduino = SimuladorArduino(seed=42)
    meteo   = SimuladorOpenWeatherMap(seed=42)
    logger  = Logger(CONFIG["log_path"])
    estado = {
        "dap": DAP_INICIAL,
        "tensao_ontem": 18.0,
        "irrigou_ontem": False,
        "chuva_ontem_mm": 0.0,
    }
    print(f"\n[Simulação] {N_DIAS} dias | "
          f"DAP {DAP_INICIAL}→{DAP_INICIAL+N_DIAS-1} | "
          f"Início: {DATA_INICIO}\n")
    print(f"{'Data':<12} {'DAP':>4} {'Tensão':>7} {'Δ':>6} "
          f"{'Chuva3d':>7} {'Tmax3d':>7} {'Classe':<15} "
          f"{'Cons%':>5} {'Dose':>5} {'Sol(s)':>6} {'Regras':>6}")
    print("-"*95)
    for dia in range(N_DIAS):
        data_atual = DATA_INICIO + timedelta(days=dia)
        try:
            estado = loop_diario(
                data=data_atual,
                estado=estado,
                ensemble=ensemble,
                arduino=arduino,
                meteo=meteo,
                logger=logger,
                config=CONFIG,
                verbose=True
            )
        except Exception as e:
            print(f"  [ERRO] {data_atual}: {e}")
            estado["dap"] += 1
            continue
    gerar_relatorio(CONFIG["log_path"], N_DIAS, CONFIG)

if __name__ == "__main__":
    main()
