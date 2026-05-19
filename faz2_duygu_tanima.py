"""
BIL216 - Emo-Challenge 2026
FAZ 2: TURBO - Hiperparametre Optimizasyonu ve Boosting Modelleri
Grup 05
"""

import os
import numpy as np
import librosa
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# =============================================
# AYAR: Dataset'in bulunduğu ANA klasör
# =============================================
VERI_KLASORU = r"C:\Users\omenh\OneDrive\Desktop\Midterm_Dataset_2026" 
# =============================================

DUYGU_ETIKETLERI = ['Notr', 'Mutlu', 'Ofkeli', 'Uzgun', 'Saskin']
YAS_GRUBU_MAP = {'C': 'Çocuk', 'K': 'Kadın', 'E': 'Erkek'}

def etiket_cikar(dosya_adi):
    for duygu in DUYGU_ETIKETLERI:
        if duygu in dosya_adi: return duygu
    return None

def metadata_cikar(dosya_adi):
    parcalar = dosya_adi.split('_')
    try:
        return parcalar[0], parcalar[1], YAS_GRUBU_MAP.get(parcalar[2], '?'), parcalar[3]
    except IndexError:
        return '?', '?', '?', '?'

def ozitelik_cikar_faz2(dosya_yolu):
    try:
        y, sr = librosa.load(dosya_yolu, duration=5, sr=22050)
        if len(y) == 0: return None

        # 20 MFCC + Delta + Delta-Delta
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        mfcc_features = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1),
            np.mean(mfcc_delta2, axis=1), np.std(mfcc_delta2, axis=1)
        ])

        # Zaman ve Temel Spektral Özellikler
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        basic_features = np.array([
            np.mean(zcr), np.std(zcr),
            np.mean(rms), np.std(rms),
            np.mean(spec_cent), np.std(spec_cent),
            np.mean(spec_roll), np.std(spec_roll)
        ])
        chroma_features = np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)])

        # Spektral Kontrast ve Tonnetz
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        
        new_spectral_features = np.concatenate([
            np.mean(spec_contrast, axis=1), np.std(spec_contrast, axis=1),
            np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1)
        ])

        # İstatistiksel Özellikler (Kurtosis & Skewness)
        signal_skew = stats.skew(y)
        signal_kurtosis = stats.kurtosis(y)
        statistical_features = np.array([signal_skew, signal_kurtosis])

        return np.concatenate([mfcc_features, basic_features, chroma_features, new_spectral_features, statistical_features])
    except:
        return None

def veri_yukle(ana_klasor):
    X, y = [], []
    alt_klasorler = [os.path.join(ana_klasor, d) for d in os.listdir(ana_klasor) 
                     if os.path.isdir(os.path.join(ana_klasor, d)) and (d.startswith('GROUP') or d.startswith('GRUP'))]
    
    tum_dosyalar = []
    for klasor in alt_klasorler:
        for dosya in os.listdir(klasor):
            if dosya.lower().endswith('.wav'): tum_dosyalar.append(os.path.join(klasor, dosya))

    print(f"Toplam {len(tum_dosyalar)} ses dosyası üzerinde Faz 2 turbo işlemler başlatılıyor...")
    for dosya_yolu in tum_dosyalar:
        dosya_adi = os.path.basename(dosya_yolu)
        etiket = etiket_cikar(dosya_adi)
        if etiket is None: continue
        ozellik = ozitelik_cikar_faz2(dosya_yolu)
        if ozellik is not None:
            X.append(ozellik)
            y.append(etiket)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = veri_yukle(VERI_KLASORU)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1] SVM için En İyi Parametreler Aranıyor (GridSearch)...")
    svm_param_grid = {
        'C': [1, 10, 50, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    }
    grid_svm = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42), svm_param_grid, cv=5, n_jobs=-1)
    grid_svm.fit(X_train_scaled, y_train)
    svm_best = grid_svm.best_estimator_
    svm_pred = svm_best.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"-> Optimize SVM Doğruluğu: %{svm_acc * 100:.2f}")

    print("\n[2] XGBoost Modeli Eğitiliyor...")
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='mlogloss')
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"-> XGBoost Doğruluğu: %{xgb_acc * 100:.2f}")

    print("\n[3] LightGBM Modeli Eğitiliyor...")
    lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, verbosity=-1)
    lgb.fit(X_train, y_train)
    lgb_pred = lgb.predict(X_test)
    lgb_acc = accuracy_score(y_test, lgb_pred)
    print(f"-> LightGBM Doğruluğu: %{lgb_acc * 100:.2f}")

    # En yüksek skoru bul ve raporla
    skorlar = {'SVM': svm_acc, 'XGBoost': xgb_acc, 'LightGBM': lgb_acc}
    en_iyi_model_adi = max(skorlar, key=skorlar.get)
    en_iyi_skor = skorlar[en_iyi_model_adi]
    y_pred = svm_pred if en_iyi_model_adi == 'SVM' else (xgb_pred if en_iyi_model_adi == 'XGBoost' else lgb_pred)

    print("\n" + "=" * 60)
    print(f"🔥 YENİ FAZ 2 NİHAİ ACCURACY: %{en_iyi_skor * 100:.2f} ({en_iyi_model_adi} ile)")
    print("=" * 60)
    
    print("\nDetaylı Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='rocket_r', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Faz 2 Turbo - Karışıklık Matrisi ({en_iyi_model_adi})')
    plt.tight_layout()
    plt.savefig('faz2_confusion_matrix.png', dpi=150)
    print("'faz2_confusion_matrix.png' dosyası güncellendi.")