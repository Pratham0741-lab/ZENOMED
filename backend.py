"""
Zenomed v3 — Flask Backend (Streamlit-Free)
RIFT 2026 Hackathon
"""

import json, time, datetime, hashlib, io, os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────
TARGET_GENES = ["CYP2D6", "CYP2C19", "CYP2C9", "SLCO1B1", "TPMT", "DPYD"]
MAX_FILE_SIZE_MB = 5  # Feature: 5MB safety limit

PHENOTYPE_CODES = {
    "Poor Metabolizer":         "PM",
    "Intermediate Metabolizer": "IM",
    "Normal Metabolizer":       "NM",
    "Rapid Metabolizer":        "RM",
    "Ultrarapid Metabolizer":   "URM",
    "Poor Function":            "PM",
    "Normal Function":          "NM",
    "Unknown":                  "Unknown",
}

STRONG_INHIBITORS = {
    "CYP2D6": {"BUPROPION","FLUOXETINE","PAROXETINE","QUINIDINE","TERBINAFINE","CINACALCET","MIRABEGRON"},
    "CYP2C19": {"FLUCONAZOLE","FLUVOXAMINE","TICLOPIDINE","VORICONAZOLE","OMEPRAZOLE"},
}

BASE_RISK_RANGE = {
    "Normal Metabolizer":       (5,  15),
    "Normal Function":          (5,  15),
    "Rapid Metabolizer":        (5,  15),
    "Intermediate Metabolizer": (35, 55),
    "Poor Metabolizer":         (85, 99),
    "Ultrarapid Metabolizer":   (82, 97),
    "Poor Function":            (80, 96),
    "Unknown":                  (40, 60),
}

DRUG_NTI_WEIGHT = {
    "Warfarin":10,"5-Fluorouracil":8,"Capecitabine":8,"Phenytoin":7,
    "Azathioprine":6,"6-Mercaptopurine":6,"Thioguanine":5,"Voriconazole":5,
    "Simvastatin":4,"Clopidogrel":4,"Repaglinide":4,"Codeine":3,
    "Tramadol":3,"Tamoxifen":3,"Metoprolol":2,"Losartan":2,
    "Atorvastatin":2,"Venlafaxine":2,"Omeprazole":1,"Escitalopram":1,"NSAIDs":2,
}

SAFE_PHENOTYPES = {"Normal Metabolizer", "Normal Function", "Rapid Metabolizer"}

DIPLOTYPE_PHENOTYPE_MAP = {
    "*1/*1":"Normal Metabolizer","*1/*2":"Normal Metabolizer","*1/*4":"Intermediate Metabolizer",
    "*4/*4":"Poor Metabolizer","*1/*41":"Intermediate Metabolizer","*2/*2":"Normal Metabolizer",
    "*1/*1xN":"Ultrarapid Metabolizer","*2/*2xN":"Ultrarapid Metabolizer","*1/*17":"Rapid Metabolizer",
    "*17/*17":"Ultrarapid Metabolizer","*1/*3":"Intermediate Metabolizer","*2/*3A":"Intermediate Metabolizer",
    "*3/*3":"Poor Metabolizer","*3A/*3A":"Poor Metabolizer","*1/*5":"Poor Function",
    "*5/*5":"Poor Function","*15/*15":"Poor Function","*2A/*2A":"Poor Metabolizer","*1/*2A":"Intermediate Metabolizer",
}

CPIC_ENGINE = {
    "CYP2D6": {
        "Poor Metabolizer": {
            "Codeine":("Ineffective","CYP2D6 Poor Metabolizers cannot O-demethylate codeine to morphine. Avoid — use non-opioid analgesics or non-CYP2D6-substrate opioids."),
            "Tramadol":("Ineffective","Impaired conversion to active M1 metabolite → reduced analgesia. Consider an alternative opioid."),
            "Tamoxifen":("Ineffective","Reduced endoxifen levels → diminished efficacy. Consider aromatase inhibitor if postmenopausal."),
            "Metoprolol":("Toxic","Beta-blocker accumulation — reduce dose 50% and monitor HR/BP closely."),
            "Venlafaxine":("Adjust Dosage","Reduced clearance increases SNRI exposure. Start at lowest dose."),
        },
        "Ultrarapid Metabolizer": {
            "Codeine":("Toxic","Ultra-rapid conversion to morphine → opioid toxicity (respiratory depression). Contraindicated."),
            "Tramadol":("Toxic","High M1 exposure. Avoid — use a non-CYP2D6-substrate analgesic."),
            "Tamoxifen":("Safe","Higher endoxifen may enhance efficacy. Standard monitoring recommended."),
            "Metoprolol":("Ineffective","Rapid clearance reduces beta-blockade. Switch to alternative beta-blocker."),
        },
        "Intermediate Metabolizer": {
            "Codeine":("Adjust Dosage","Partial prodrug conversion → reduced analgesia. Titrate carefully."),
            "Tamoxifen":("Adjust Dosage","Reduced endoxifen — consider dose escalation or switch."),
            "Metoprolol":("Safe","Mildly reduced clearance. Standard dosing with routine monitoring."),
        },
        "Normal Metabolizer": {
            "Codeine":("Safe","Standard dosing is appropriate."),
            "Tramadol":("Safe","Standard dosing is appropriate."),
            "Tamoxifen":("Safe","Standard dosing is appropriate."),
            "Metoprolol":("Safe","Standard dosing is appropriate."),
            "Venlafaxine":("Safe","Standard dosing is appropriate."),
        },
    },
    "CYP2C19": {
        "Poor Metabolizer": {
            "Clopidogrel":("Ineffective","Severely reduced prodrug activation → inadequate antiplatelet effect. Use prasugrel or ticagrelor."),
            "Omeprazole":("Toxic","PPI accumulation → excessive acid suppression. Reduce dose or switch PPI."),
            "Escitalopram":("Adjust Dosage","Reduced clearance → elevated SSRI exposure. Start at 50% dose."),
            "Voriconazole":("Toxic","Antifungal overexposure → hepatotoxicity/visual disturbances. Monitor levels."),
        },
        "Ultrarapid Metabolizer": {
            "Clopidogrel":("Safe","Enhanced activation; standard antiplatelet inhibition."),
            "Omeprazole":("Ineffective","Rapid clearance → inadequate acid suppression. Increase dose or switch."),
            "Escitalopram":("Ineffective","Rapid clearance reduces therapeutic effect. Consider dose increase."),
        },
        "Intermediate Metabolizer": {
            "Clopidogrel":("Adjust Dosage","Partial reduction in antiplatelet effect. Consider prasugrel in high-risk ACS."),
            "Escitalopram":("Adjust Dosage","Moderate exposure increase. Start at lower dose range."),
            "Omeprazole":("Safe","Slightly reduced exposure. Standard dosing appropriate."),
        },
        "Normal Metabolizer": {
            "Clopidogrel":("Safe","Standard dosing is appropriate."),
            "Omeprazole":("Safe","Standard dosing is appropriate."),
            "Escitalopram":("Safe","Standard dosing is appropriate."),
        },
    },
    "CYP2C9": {
        "Poor Metabolizer": {
            "Warfarin":("Toxic","Impaired S-warfarin metabolism → dramatically elevated bleeding risk. Reduce initial dose 50–75%; monitor INR closely."),
            "Phenytoin":("Toxic","Phenytoin toxicity (nystagmus, ataxia). Start 25–50% dose with TDM."),
            "NSAIDs":("Toxic","Celecoxib/ibuprofen accumulation → elevated GI/CV risk. Use acetaminophen."),
            "Losartan":("Adjust Dosage","Reduced E3174 conversion. Consider valsartan or another ARB."),
        },
        "Intermediate Metabolizer": {
            "Warfarin":("Adjust Dosage","Moderate reduction in clearance — reduce initiation dose 20–40%."),
            "Phenytoin":("Adjust Dosage","Moderate toxicity risk — use therapeutic drug monitoring."),
            "NSAIDs":("Safe","Mildly reduced clearance. Standard dosing with GI monitoring."),
        },
        "Normal Metabolizer": {
            "Warfarin":("Safe","Standard dosing is appropriate."),
            "Phenytoin":("Safe","Standard dosing is appropriate."),
            "NSAIDs":("Safe","Standard dosing is appropriate."),
        },
    },
    "SLCO1B1": {
        "Poor Function": {
            "Simvastatin":("Toxic","Reduced OATP1B1 function → statin plasma accumulation → myopathy/rhabdomyolysis. Switch to pravastatin or use low-dose rosuvastatin."),
            "Atorvastatin":("Adjust Dosage","Moderate myopathy risk. Limit to 20 mg/day and monitor CK."),
            "Repaglinide":("Toxic","Elevated plasma exposure → hypoglycemia risk. Reduce dose and monitor glucose."),
        },
        "Normal Function": {
            "Simvastatin":("Safe","Standard dosing is appropriate."),
            "Atorvastatin":("Safe","Standard dosing is appropriate."),
            "Repaglinide":("Safe","Standard dosing is appropriate."),
        },
    },
    "TPMT": {
        "Poor Metabolizer": {
            "Azathioprine":("Toxic","TPMT deficiency → 6-TGN accumulation → severe myelosuppression. Use 10% dose or switch to mycophenolate."),
            "6-Mercaptopurine":("Toxic","High 6-TGN → leukopenia/agranulocytosis. Drastic dose reduction mandatory."),
            "Thioguanine":("Toxic","Profound myelosuppression. Dose reduction >10× required."),
        },
        "Intermediate Metabolizer": {
            "Azathioprine":("Adjust Dosage","Partial TPMT — start at 30–70% dose with CBC monitoring."),
            "6-Mercaptopurine":("Adjust Dosage","Reduce initial dose ~50%; adjust based on CBC and tolerability."),
        },
        "Normal Metabolizer": {
            "Azathioprine":("Safe","Standard dosing is appropriate."),
            "6-Mercaptopurine":("Safe","Standard dosing is appropriate."),
        },
    },
    "DPYD": {
        "Poor Metabolizer": {
            "5-Fluorouracil":("Toxic","DPD deficiency → severe 5-FU toxicity: mucositis, myelosuppression, neurotoxicity. Contraindicated."),
            "Capecitabine":("Toxic","Capecitabine is a 5-FU prodrug — identical DPYD toxicity risk. Contraindicated."),
        },
        "Intermediate Metabolizer": {
            "5-Fluorouracil":("Adjust Dosage","Partial DPD — start at 50% dose; titrate based on tolerability."),
            "Capecitabine":("Adjust Dosage","Start at 50% dose. Increase cautiously based on clinical response."),
        },
        "Normal Metabolizer": {
            "5-Fluorouracil":("Safe","Standard dosing is appropriate."),
            "Capecitabine":("Safe","Standard dosing is appropriate."),
        },
    },
}

SAMPLE_VCF = """\
##fileformat=VCFv4.2
##fileDate=20260101
##source=PharmaGuardSimulator_v1.0
##reference=GRCh38
##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">
##INFO=<ID=STAR,Number=1,Type=String,Description="Star allele diplotype">
##INFO=<ID=RS,Number=1,Type=String,Description="dbSNP rsID">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr22\t42526694\trs3892097\tC\tT\t100\tPASS\tGENE=CYP2D6;STAR=*4/*4;RS=3892097;AF=0.07
chr10\t96741053\trs4244285\tG\tA\t100\tPASS\tGENE=CYP2C19;STAR=*2/*2;RS=4244285;AF=0.03
chr10\t96702047\trs1799853\tC\tT\t100\tPASS\tGENE=CYP2C9;STAR=*1/*3;RS=1799853;AF=0.12
chr12\t21332628\trs4149056\tT\tC\t100\tPASS\tGENE=SLCO1B1;STAR=*1/*5;RS=4149056;AF=0.15
chr6\t18143955\trs1142345\tT\tC\t100\tPASS\tGENE=TPMT;STAR=*1/*3A;RS=1142345;AF=0.05
chr1\t97981395\trs3918290\tC\tT\t100\tPASS\tGENE=DPYD;STAR=*1/*2A;RS=3918290;AF=0.01
"""

# ─────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def check_phenoconversion(gene, concomitant_meds):
    inhibitors = STRONG_INHIBITORS.get(gene, set())
    for med in concomitant_meds:
        if med.strip().upper() in inhibitors:
            return True, med.strip().upper()
    return False, None

def parse_concomitant_meds(raw):
    if not raw or not raw.strip():
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]

def compute_risk_percentage(phenotype, drug, gene):
    lo, hi = BASE_RISK_RANGE.get(phenotype, (40, 60))
    seed_str = f"{gene}|{drug}|{phenotype}"
    jitter = int(hashlib.md5(seed_str.encode()).hexdigest()[:4], 16) % 6
    base_pct = lo + (jitter / 5.0) * (hi - lo)
    if phenotype not in SAFE_PHENOTYPES:
        base_pct += DRUG_NTI_WEIGHT.get(drug, 0)
    return round(min(max(base_pct, 0.0), 100.0), 1)

def severity_from_pct(pct):
    if pct < 20: return "none"
    elif pct < 40: return "low"
    elif pct < 60: return "moderate"
    elif pct < 80: return "high"
    else: return "critical"

def infer_phenotype(gene, diplotype):
    """
    Graceful handling of missing annotations in diplotypes.
    """
    if not diplotype or diplotype == ".":
        return "Unknown"
    if diplotype in DIPLOTYPE_PHENOTYPE_MAP:
        return DIPLOTYPE_PHENOTYPE_MAP[diplotype]
    null_alleles = {"*4","*5","*6","*7","*8","*11","*2A"}
    gain_alleles = {"*1xN","*2xN","*17"}
    parts = [p.strip() for p in diplotype.split("/")]
    if len(parts) < 2: return "Normal Metabolizer" 
    null_count = sum(1 for p in parts if p in null_alleles)
    gain_count = sum(1 for p in parts if any(g in p for g in gain_alleles))
    if gain_count >= 1: return "Ultrarapid Metabolizer"
    if null_count == 2: return "Poor Metabolizer"
    if null_count == 1: return "Intermediate Metabolizer"
    return "Normal Metabolizer"

def parse_vcf(file_content):
    """
    Enhanced with clear error messages and validation for VCF structure.
    """
    lines = file_content.splitlines()
    if not any(line.strip().startswith("#CHROM") for line in lines):
        raise ValueError("Invalid VCF Format: The file is missing the mandatory #CHROM header row.")
        
    meta = {"total_lines":len(lines),"metadata_lines":0,"total_variants":0,
            "filtered_variants":0,"parsing_success":False,"column_map":{}}
    col_map = {}
    variants = []
    
    for line_num, line in enumerate(lines, 1):
        try:
            line = line.strip()
            if not line: continue
            if line.startswith("##"):
                meta["metadata_lines"] += 1
                continue
            if line.startswith("#CHROM"):
                headers = line.lstrip("#").split("\t")
                col_map = {h.upper(): i for i, h in enumerate(headers)}
                if "CHROM" not in col_map or "POS" not in col_map:
                    raise ValueError("Incomplete VCF Columns: CHROM and POS are required.")
                meta["column_map"] = col_map
                continue
            if not col_map: continue
            
            fields = line.split("\t")
            meta["total_variants"] += 1
            
            def safe_get(col_name, default="."):
                idx = col_map.get(col_name)
                return fields[idx] if idx is not None and idx < len(fields) else default
                
            chrom = safe_get("CHROM")
            pos = safe_get("POS")
            id_col = safe_get("ID")
            ref = safe_get("REF")
            alt = safe_get("ALT")
            info_raw = safe_get("INFO")
            
            info = {}
            for tag in info_raw.split(";"):
                if "=" in tag:
                    parts = tag.split("=", 1)
                    info[parts[0].strip().upper()] = parts[1].strip()
                else:
                    info[tag.strip().upper()] = "true"
            
            gene = info.get("GENE", "").upper()
            star = info.get("STAR", "")
            if not gene: continue
            
            rs_info = info.get("RS", "")
            rsid = id_col if id_col not in (".", "", None) else (f"rs{rs_info}" if rs_info else "unknown")
            
            if gene not in [g.upper() for g in TARGET_GENES]:
                continue
                
            meta["filtered_variants"] += 1
            variants.append({"chrom":chrom,"pos":pos,"rsid":rsid,"ref":ref,"alt":alt,
                             "gene":gene,"diplotype":star,"info_raw":info_raw})
        except Exception as e:
            continue

    meta["parsing_success"] = meta["total_variants"] > 0 or bool(variants)
    return variants, meta

def run_risk_engine(variants, concomitant_meds=None):
    results = []
    seen_genes = set()
    concomitant_meds = concomitant_meds or []
    for v in variants:
        gene = v["gene"]
        if gene in seen_genes: continue
        seen_genes.add(gene)
        genetic_phenotype = infer_phenotype(gene, v["diplotype"])
        phenoconverted, inhibitor_drug = check_phenoconversion(gene, concomitant_meds)
        effective_phenotype = "Poor Metabolizer" if phenoconverted else genetic_phenotype
        gene_data = CPIC_ENGINE.get(gene, {})
        drug_map = gene_data.get(effective_phenotype, gene_data.get("Normal Metabolizer", {}))
        for drug, (risk_label, recommendation) in drug_map.items():
            pct = compute_risk_percentage(effective_phenotype, drug, gene)
            full_rec = recommendation
            if phenoconverted:
                full_rec = f"⚠️ Phenoconversion by {inhibitor_drug}: effective phenotype is Poor Metabolizer regardless of genotype. — {recommendation}"
            results.append({
                "gene": gene, "drug": drug,
                "diplotype": v["diplotype"] or "Not specified",
                "genetic_phenotype": genetic_phenotype,
                "phenotype": effective_phenotype,
                "phenoconversion_detected": phenoconverted,
                "phenoconversion_drug": inhibitor_drug or "",
                "risk": risk_label, "risk_pct": pct,
                "severity": severity_from_pct(pct),
                "recommendation": full_rec,
                "rsid": v["rsid"], "chrom": v["chrom"], "pos": v["pos"],
            })
    return results

def gemini_clinical_explanation(api_key, rsid, gene, drug, risk, risk_pct):
    if not api_key:
        return (f"[Add Gemini API key for AI explanations] Variant {rsid} in {gene} produces a "
                f"{risk_pct}% quantitative risk score for {drug} ({risk}). This reflects disrupted enzymatic metabolism.")
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (f"You are a clinical pharmacogenomics expert. In 3-4 concise sentences, explain the "
                  f"pharmacokinetic/pharmacodynamic mechanism by which variant {rsid} in gene {gene} "
                  f"produces a quantitative risk score of {risk_pct}% for the drug {drug}, classified as '{risk}'. "
                  f"Be precise but accessible to a clinical practitioner.")
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        return f"AI explanation error: {str(e)[:200]}"

def gemini_patient_literacy_summary(api_key, meta, variants):
    if not api_key:
        return (f"Your genetic file was successfully read. It contained {meta.get('total_variants',0)} total "
                f"genetic markers, of which {meta.get('filtered_variants',0)} are relevant to how your body "
                "processes certain medications. Please discuss these results with your doctor or pharmacist.")
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        gene_list = list({v["gene"] for v in variants})
        prompt = (f"You are a compassionate patient educator. Explain these genetic test results in simple, "
                  f"reassuring language for a patient with no medical background. The file had "
                  f"{meta.get('total_variants',0)} genetic variants; {meta.get('filtered_variants',0)} are in "
                  f"pharmacogenomically important genes: {', '.join(gene_list) if gene_list else 'none'}. "
                  "Write 4-5 sentences without jargon. Emphasize that a doctor should be consulted.")
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        return f"Summary error: {str(e)[:200]}"

def gemini_wellness_tips(api_key, genes):
    if not api_key:
        return ("• Support liver metabolism: cruciferous vegetables (broccoli, Brussels sprouts)\n"
                "• Avoid grapefruit — inhibits CYP3A4 and interacts with many drugs\n"
                "• Limit alcohol: competes with hepatic enzymes, depletes glutathione\n"
                "• Leafy greens support vitamin K stability (important for warfarin users)\n"
                "• Stay hydrated to support renal drug clearance\n"
                "• Moderate aerobic exercise improves hepatic blood flow and drug metabolism")
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (f"You are an integrative medicine consultant. A patient has pharmacogenomic variants in: {', '.join(genes)}. "
                  "Provide 5-7 evidence-based lifestyle and dietary recommendations. Use bullet points. 1-2 sentences each.")
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        return f"Wellness tips error: {str(e)[:200]}"

def generate_historical_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    phenotype_weights = {
        "CYP2D6":[0.07,0.20,0.60,0.08,0.05],"CYP2C19":[0.03,0.26,0.63,0.05,0.03],
        "CYP2C9":[0.04,0.25,0.68,0.02,0.01],"SLCO1B1":[0.15,0.30,0.50,0.03,0.02],
        "TPMT":[0.003,0.11,0.88,0.005,0.002],"DPYD":[0.02,0.10,0.85,0.02,0.01],
    }
    phenotype_labels = ["Poor Metabolizer","Intermediate Metabolizer","Normal Metabolizer","Rapid Metabolizer","Ultrarapid Metabolizer"]
    risk_encoding = {"Safe":0,"Adjust Dosage":1,"Toxic":2,"Ineffective":3,"Unknown":4}
    records = []
    for _ in range(n):
        gene = rng.choice(TARGET_GENES)
        w = np.array(phenotype_weights[gene], dtype=float); w /= w.sum()
        pheno_idx = int(rng.choice(len(phenotype_labels), p=w))
        pheno = phenotype_labels[pheno_idx]
        if "Poor" in pheno: risk = rng.choice(["Toxic","Ineffective","Adjust Dosage"],p=[0.40,0.40,0.20])
        elif "Ultrarapid" in pheno: risk = rng.choice(["Toxic","Ineffective","Safe"],p=[0.35,0.35,0.30])
        elif "Intermediate" in pheno: risk = rng.choice(["Adjust Dosage","Safe","Unknown"],p=[0.60,0.30,0.10])
        else: risk = rng.choice(["Safe","Adjust Dosage"],p=[0.90,0.10])
        records.append({"gene":gene,"phenotype":pheno,"phenotype_score":pheno_idx,"risk_label":risk,
                        "risk_score":risk_encoding[risk],"month":int(rng.integers(1,13)),"allele_freq":float(rng.beta(2,8))})
    return pd.DataFrame(records)

def run_kmeans_clustering(df, n_clusters=4):
    features = df[["phenotype_score","risk_score","allele_freq"]].copy()
    scaled = StandardScaler().fit_transform(features)
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
    df["cluster"] = km.fit_predict(scaled)
    cluster_risk = df.groupby("cluster")["risk_score"].mean()
    df["cluster_label"] = df["cluster"].apply(lambda c: "High-Risk Cluster" if cluster_risk[c] >= 2.0 else "Standard-Risk Cluster")
    return df

def build_json_report(patient_id, result, meta, ai_explanation):
    return {
        "patient_id": patient_id,
        "drug": result["drug"],
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "risk_assessment": {
            "risk_label": result["risk"],
            "confidence_score": float(result["risk_pct"]),
            "severity": result["severity"]
        },
        "pharmacogenomic_profile": {
            "primary_gene": result["gene"],
            "diplotype": result["diplotype"],
            "phenotype": PHENOTYPE_CODES.get(result["phenotype"], "Unknown"),
            "detected_variants": [
                {
                    "rsid": result["rsid"],
                    "chrom": result.get("chrom", ""),
                    "pos": result.get("pos", "")
                }
            ]
        },
        "clinical_recommendation": {
            "guideline_text": result["recommendation"],
            "phenoconversion_detected": result.get("phenoconversion_detected", False),
            "phenoconversion_drug": result.get("phenoconversion_drug", "")
        },
        "llm_generated_explanation": {
            "summary": ai_explanation
        },
        "quality_metrics": {
            "vcf_parsing_success": meta.get("parsing_success", False),
            "total_variants_parsed": meta.get("total_variants", 0)
        }
    }

# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        patient_id = request.form.get("patient_id", "PATIENT-001")
        gemini_key = request.form.get("gemini_key", "")
        concomitant_raw = request.form.get("concomitant_meds", "")
        concomitant_meds = parse_concomitant_meds(concomitant_raw)

        vcf_file = request.files.get("vcf_file")
        if vcf_file:
            # Feature: Enforce file size limit
            vcf_file.seek(0, os.SEEK_END)
            file_size = vcf_file.tell()
            vcf_file.seek(0)
            
            if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                return jsonify({
                    "error": "File Too Large",
                    "user_hint": f"The genomic file exceeds the {MAX_FILE_SIZE_MB}MB limit. Please upload a smaller VCF."
                }), 413
                
            vcf_content = vcf_file.read().decode("utf-8", errors="replace")
        else:
            vcf_content = SAMPLE_VCF

        try:
            variants, meta = parse_vcf(vcf_content)
        except ValueError as ve:
            return jsonify({"error": str(ve), "user_hint": "Please ensure your file is a valid .vcf or .txt and contains a #CHROM header."}), 400

        if not variants:
            return jsonify({
                "error": "No Relevant Variants Found",
                "user_hint": f"Your VCF was parsed but contains no target PGx genes ({', '.join(TARGET_GENES)}). Ensure your file includes these specific annotations."
            }), 400

        results = run_risk_engine(variants, concomitant_meds)
        overall_risk_pct = round(sum(r["risk_pct"] for r in results) / len(results), 1) if results else 0
        literacy_summary = gemini_patient_literacy_summary(gemini_key, meta, variants)
        wellness = gemini_wellness_tips(gemini_key, list({v["gene"] for v in variants}))

        reports = []
        for r in results:
            ai_text = gemini_clinical_explanation(gemini_key, r["rsid"], r["gene"], r["drug"], r["risk"], r["risk_pct"])
            reports.append(build_json_report(patient_id, r, meta, ai_text))

        df_hist = generate_historical_data(n=500)
        df_clustered = run_kmeans_clustering(df_hist.copy())
        high_risk_n = int((df_clustered["cluster_label"].str.contains("High")).sum())
        monthly = df_hist.groupby(["month","risk_label"]).size().reset_index(name="count")
        trend_data = monthly.to_dict(orient="records")
        pie_data = df_hist["risk_label"].value_counts().to_dict()
        allele_data = df_hist[["gene","allele_freq"]].sample(min(200, len(df_hist)), random_state=42).to_dict(orient="records")
        cluster_data = df_clustered[["allele_freq","risk_score","cluster_label","gene"]].sample(min(300, len(df_clustered)), random_state=42).to_dict(orient="records")

        return jsonify({
            "patient_id": patient_id,
            "meta": meta,
            "overall_risk_pct": overall_risk_pct,
            "results": results,
            "reports": reports,
            "literacy_summary": literacy_summary,
            "wellness_tips": wellness,
            "analytics": {
                "high_risk_n": high_risk_n,
                "total_simulated": 500,
                "trend_data": trend_data,
                "pie_data": pie_data,
                "allele_data": allele_data,
                "cluster_data": cluster_data,
            },
            "concomitant_meds": concomitant_meds,
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": "System Error during analysis.",
            "user_hint": "An unexpected error occurred. Please verify your internet connection and Gemini API key.",
            "trace": traceback.format_exc()
        }), 500

@app.route("/api/sample-vcf", methods=["GET"])
def get_sample_vcf():
    return SAMPLE_VCF, 200, {"Content-Type": "text/plain"}

if __name__ == "__main__":
    app.run(debug=True, port=5000)