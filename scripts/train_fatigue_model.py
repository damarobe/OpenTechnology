import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import seaborn as sns
import glob
import soundfile as sf
from scipy import signal
import pandas as pd
import random
from scipy.stats import ttest_ind
from scipy.stats import ttest_ind, skew, kurtosis, entropy

# Feature extraction function
import maad
from maad import sound, rois, features
import maad.util

# Paths to directories (replace accordingly)
DATA_PATH = 'C:/Users/rober/Desktop/Klaviatūros mygtukų garsai/dataset'
SPECTROGRAM_PATH = 'C:/Users/rober/Desktop/Klaviatūros mygtukų garsai/spectrograms'

# Feature extraction function
maad_features =[]

def calculate_audio_features(file_path):
    # Load audio file (convert to mono)
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # Calculate features
    myfeatures = []

    # 1. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    myfeatures.append(spectral_centroid)

    # 2. Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    myfeatures.append(spectral_bandwidth)

    # 3. Spectral Flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y).mean()
    myfeatures.append(spectral_flatness)

    # 4. Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
    myfeatures.append(spectral_contrast)

    # 5. High-Frequency Energy Ratio (Above 5 kHz)
    high_freq_energy_ratio = np.sum(np.abs(librosa.stft(y))[int(5000/(sr/2) * (y.size / sr)):]**2) / np.sum(np.abs(librosa.stft(y))**2)
    myfeatures.append(high_freq_energy_ratio)

    # 6. Zero-Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
    myfeatures.append(zero_crossing_rate)

    # 7. RMS Energy
    rms_energy = librosa.feature.rms(y=y).mean()
    myfeatures.append(rms_energy)

    # 8. Temporal Flatness (using Variance over Time)
    temporal_flatness = np.var(y) / np.mean(y) if np.mean(y) != 0 else 0
    myfeatures.append(temporal_flatness)

    # 9. Spectral Entropy
    stft = np.abs(librosa.stft(y))**2
#    p_spectrum = stft / np.sum(stft, axis=0, keepdims=True)
#    spectral_entropy = -np.sum(p_spectrum * np.log2(p_spectrum + 1e-10)) / p_spectrum.shape[0]
#    myfeatures.append(spectral_entropy.mean())

    # 10. Peak Frequency Count
    peak_frequency_count = np.sum(np.abs(np.diff(np.sign(np.diff(y)))) == 2)
    myfeatures.append(peak_frequency_count)

    # 11. Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    myfeatures.extend(mfccs)

    # 12. Chromagram
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    myfeatures.extend(chroma)

    # 13. Tonnetz (tonal centroid features)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).mean(axis=1)
    myfeatures.extend(tonnetz)

    # 14. Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    myfeatures.append(spectral_rolloff)

    # 15. Spectral Flux
    spectral_flux = np.sqrt(np.sum(np.diff(librosa.stft(y), axis=1) ** 2, axis=0)).mean()
    myfeatures.append(spectral_flux.real)



    # 16. High-Frequency Emphasis (weighted average of high-frequency MFCCs scaled by spectral rolloff)
    high_freq_mfccs = mfccs[8:].mean(axis=0)  # MFCCs 9, 10, 11
    high_freq_emphasis = np.mean(high_freq_mfccs * spectral_rolloff)
    myfeatures.append(high_freq_emphasis)

    # 17. Dynamic MFCC Variability (Standard deviation of delta MFCCs)
    mfcc_delta = librosa.feature.delta(mfccs)
    dynamic_mfcc_variability = mfcc_delta.std()
    myfeatures.append(dynamic_mfcc_variability)

    # 18. Spectral Roll-Off Slope (using linear regression on spectral rolloff over time)
#    times = np.arange(spectral_rolloff.shape[0])
#    rolloff_slope = np.polyfit(times, spectral_rolloff.flatten(), 1)[0]  # Slope of rolloff
#    myfeatures.append(rolloff_slope)

    # 19. MFCC Mean and Variance
    for i in range(mfccs.shape[0]):
        myfeatures.append(np.mean(mfccs[i]))
        myfeatures.append(np.var(mfccs[i]))

    # 20. High-to-Low Energy Ratio (energy above vs. below spectral centroid)
#    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
#    high_energy = np.sum(stft[spectral_centroid >= sr / 2], axis=0)
#    low_energy = np.sum(stft[spectral_centroid < sr / 2], axis=0)
#    high_to_low_energy_ratio = np.mean(high_energy / (low_energy + 1e-10))  # Avoid division by zero
#    myfeatures.append(high_to_low_energy_ratio)

    # 21. Chromagram Differences (difference between specific chroma bins)
    chroma_diff = np.abs(chroma[10] - chroma[7])
    chromagram_difference = np.mean(chroma_diff)
    myfeatures.append(chromagram_difference)



    # Calculate core features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    stft = np.abs(librosa.stft(y))
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    
    # 1. Delta and Delta-Delta MFCCs (mean and std)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta_delta = librosa.feature.delta(mfccs, order=2)
    myfeatures.append(mfcc_delta.mean())
    myfeatures.append(mfcc_delta.std())
    myfeatures.append(mfcc_delta_delta.mean())
    myfeatures.append(mfcc_delta_delta.std())

    # 2. Band-Specific Energy Ratios
    freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0] * 2 - 1)
    low_freq_energy = np.sum(stft[freqs < 500, :])  # Low-frequency (<500 Hz)
    mid_freq_energy = np.sum(stft[(freqs >= 500) & (freqs < 2000), :])  # Mid-frequency (500-2000 Hz)
    high_freq_energy = np.sum(stft[freqs >= 2000, :])  # High-frequency (>2000 Hz)
    total_energy = np.sum(stft)
    myfeatures.append(low_freq_energy / total_energy)
    myfeatures.append(mid_freq_energy / total_energy)
    myfeatures.append(high_freq_energy / total_energy)

    # 3. Spectral Skewness and Kurtosis
    myfeatures.append(skew(stft.flatten()))
    myfeatures.append(kurtosis(stft.flatten()))

    # 4. Spectral Entropy
    power_spectrum = np.square(stft)
    power_spectrum_norm = power_spectrum / np.sum(power_spectrum)
    spectral_entropy = entropy(power_spectrum_norm.mean(axis=1))
    myfeatures.append(spectral_entropy)

    # 5. Zero-Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y)
    myfeatures.append(np.mean(zcr))

    # 6. High-Frequency Energy Ratio (energy > 5 kHz)
    high_freq_energy_ratio = np.sum(stft[freqs >= 5000]) / total_energy
    myfeatures.append(high_freq_energy_ratio)

    # 7. Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
    for i, contrast in enumerate(spectral_contrast):
        myfeatures.append(contrast)

    # 8. Temporal Energy Variability (variance of RMS energy)
    rms = librosa.feature.rms(y=y)
    myfeatures.append(np.var(rms))

    # 9. Spectral Roll-Off Slope (slope of roll-off over time)
    rolloff_times = np.arange(rolloff.shape[1])
    rolloff_slope = np.polyfit(rolloff_times, rolloff.flatten(), 1)[0]  # Get the slope
    myfeatures.append(rolloff_slope)

    # 10. Harmonic-to-Noise Ratio (HNR)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    hnr = np.sum(np.square(y_harmonic)) / (np.sum(np.square(y_percussive)) + 1e-10)  # Avoid division by zero
    myfeatures.append(hnr)


    return myfeatures

# Load data and preprocess
myfeatures = []
labels = []

for file_path in glob.glob(os.path.join(DATA_PATH, "*.m4a")):
    label = 0 if "NEpavargęs" in file_path else 1  # Adjust based on naming conventions or directory structure
    labels.append(label)
    audio_features = calculate_audio_features(file_path)
    print(audio_features)
    myfeatures.append(audio_features)
    
    # Extract acoustic features using maad
    y, sr = librosa.load(file_path, sr=None, mono=True)
    Sxx_power, tn, fn, ext = sound.spectrogram(y, sr)
    maad_features = features.all_temporal_alpha_indices(y, sr)
    # Append maad features to the list of features
    for key, value in maad_features.items():
        if np.isscalar(value):
            myfeatures.append(value)

X = np.array(myfeatures)
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier for Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate feature importance using Random Forest
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)


# Plot feature importance
feature_names = [
    'Spectral Centroid', 'Spectral Bandwidth', 'Spectral Flatness', 'Spectral Contrast',
    'High-Frequency Energy Ratio', 'Zero-Crossing Rate', 'RMS Energy', 'Temporal Flatness',
    'Peak Frequency Count', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5', 'MFCC 6', 'MFCC 7',
    'MFCC 8', 'MFCC 9', 'MFCC 10', 'MFCC 11', 'MFCC 12', 'MFCC 13', 'Chromagram 1', 'Chromagram 2', 'Chromagram 3',
    'Chromagram 4', 'Chromagram 5', 'Chromagram 6', 'Chromagram 7', 'Chromagram 8', 'Chromagram 9', 'Chromagram 10',
    'Chromagram 11', 'Chromagram 12', 'Tonnetz 1', 'Tonnetz 2', 'Tonnetz 3', 'Tonnetz 4', 'Tonnetz 5', 'Tonnetz 6',
    'Spectral Rolloff', 'Spectral Flux', 'High-Frequency Emphasis', 'Dynamic MFCC Variability', 
    'MFCC 1 Mean', 'MFCC 1 Variance', 'MFCC 2 Mean', 'MFCC 2 Variance', 'MFCC 3 Mean', 'MFCC 3 Variance', 'MFCC 4 Mean',
    'MFCC 4 Variance', 'MFCC 5 Mean', 'MFCC 5 Variance', 'MFCC 6 Mean', 'MFCC 6 Variance', 'MFCC 7 Mean', 'MFCC 7 Variance',
    'MFCC 8 Mean', 'MFCC 8 Variance', 'MFCC 9 Mean', 'MFCC 9 Variance', 'MFCC 10 Mean', 'MFCC 10 Variance',
    'MFCC 11 Mean', 'MFCC 11 Variance', 'MFCC 12 Mean', 'MFCC 12 Variance', 'MFCC 13 Mean', 'MFCC 13 Variance',
    'Chromagram Difference 10-7', 
    'MFCC Delta Mean', 'MFCC Delta Std', 'MFCC DeltaDelta Mean', 'MFCC DeltaDelta Std',
    'Low Freq Energy Ratio', 'Mid Freq Energy Ratio', 'High Freq Energy Ratio',
    'Spectral Skewness', 'Spectral Kurtosis', 'Spectral Entropy',
    'Zero Crossing Rate', 'High Frequency Energy Ratio',
    'Spectral Contrast Band 1', 'Spectral Contrast Band 2', 'Spectral Contrast Band 3',
    'Spectral Contrast Band 4', 'Spectral Contrast Band 5', 'Spectral Contrast Band 6',
    'Spectral Contrast Band 7', 'RMS Energy Variance', 'Spectral Roll-Off Slope',
    'Harmonic-to-Noise Ratio'
]

# Add MAAD feature names
def update_feature_names_with_maad(maad_features):
    for key in maad_features.keys():
        if key not in feature_names:
            feature_names.append(key)

SPECTRAL_FEATURES=['MEANf','VARf','SKEWf','KURTf','NBPEAKS','LEQf',
    'ENRf','BGNf','SNRf','Hf', 'EAS','ECU','ECV','EPS','EPS_KURT','EPS_SKEW','ACI',
    'NDSI','rBA','AnthroEnergy','BioEnergy','BI','ROU','ADI','AEI','LFC','MFC','HFC',
    'ACTspFract','ACTspCount','ACTspMean', 'EVNspFract','EVNspMean','EVNspCount',
    'TFSD','H_Havrda','H_Renyi','H_pairedShannon', 'H_gamma', 'H_GiniSimpson','RAOQ',
    'AGI','ROItotal','ROIcover'
    ]

TEMPORAL_FEATURES=['ZCR','MEANt', 'VARt', 'SKEWt', 'KURTt',
    'LEQt','BGNt', 'SNRt','MED', 'Ht','ACTtFraction', 'ACTtCount',
    'ACTtMean','EVNtFraction', 'EVNtMean', 'EVNtCount'
    ]

update_feature_names_with_maad(maad_features)

# Plot feature importance for top 10 most important features
important_features_top_10 = np.argsort(importances)[::-1][:10]
plt.figure(figsize=(10, 6))
plt.barh(range(10), importances[important_features_top_10])
plt.yticks(range(10), [feature_names[i] for i in important_features_top_10])
plt.xlabel("Feature Importance")
plt.title("Top 10 Feature Importance Evaluation using Random Forest")
plt.tight_layout()
plt.savefig('feature_importance.png', format='png', dpi=300)
#plt.show()


# Statistical plot to show difference between feature values between two classes for top 10 most important features
class_0_features = X[y == 0]
class_1_features = X[y == 1]

important_features_top_10 = np.argsort(importances)[::-1][:10]
p_values = []

fig, axs = plt.subplots(5, 2, figsize=(15, 20))
axs = axs.ravel()

for idx, feature_idx in enumerate(important_features_top_10):
    _, p_value = ttest_ind(class_0_features[:, feature_idx], class_1_features[:, feature_idx], equal_var=False)
    p_values.append(p_value)
    sns.kdeplot(class_0_features[:, feature_idx], ax=axs[idx], label='Class 0', shade=True)
    sns.kdeplot(class_1_features[:, feature_idx], ax=axs[idx], label='Class 1', shade=True)
    axs[idx].set_title(f'{feature_names[feature_idx]} (p-value: {p_values[idx]:.3e})')
    axs[idx].legend()

plt.tight_layout()
plt.savefig('kernel_density.png', format='png', dpi=300)
#plt.show()



# Train classifier using only important features
important_features = np.argsort(importances)[::-1][:5]
X_train_selected = X_train[:, important_features]
X_test_selected = X_test[:, important_features]


from xgboost import XGBClassifier

xgb_classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_classifier.fit(X_train_selected, y_train)

# Performance evaluation
y_pred = xgb_classifier.predict(X_test_selected)


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', format='png', dpi=300)
#plt.show()

# Comprehensive Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [accuracy, precision, recall, f1]
}

metrics_df = pd.DataFrame(metrics_data)
print(metrics_df)

# Plotting the metrics table
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')
ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
plt.title('Classifier Performance Metrics')
plt.savefig('performance_metrics.png', format='png', dpi=300)
#plt.show()
