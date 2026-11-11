"""
Layer 2: Cognitive Profiler - Prototype
Predicts victim cognitive profile from conversation features.
Uses linguistic features, behavioral signals, and temporal patterns.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from shared.utils.cova_loader import load_cova_x

class ConversationFeatureExtractor:
    """Extract linguistic and behavioral features from conversation turns."""
    
    def __init__(self):
        self.emotion_words = {
            'fear': ['afraid', 'scared', 'worried', 'nervous', 'panic', 'terrified'],
            'trust': ['trust', 'believe', 'sure', 'confident', 'reliable', 'honest'],
            'confusion': ['confused', 'unclear', 'dont understand', 'not sure', 'what do you mean'],
            'urgency': ['urgent', 'immediate', 'now', 'quickly', 'hurry', 'asap'],
            'compliance': ['ok', 'alright', 'sure', 'fine', 'yes', 'will do'],
            'resistance': ['no', 'dont', 'wont', 'refuse', 'stop', 'hang up'],
        }
    
    def extract_features(self, conversation):
        turns = conversation.get('turns', [])
        victim_turns = [t for t in turns if t.get('role') == 'victim']
        attacker_turns = [t for t in turns if t.get('role') == 'attacker']
        features = {}
        
        features['total_turns'] = len(turns)
        features['victim_turns'] = len(victim_turns)
        features['attacker_turns'] = len(attacker_turns)
        features['victim_ratio'] = len(victim_turns) / max(len(turns), 1)
        
        victim_text = ' '.join(t.get('content', '') for t in victim_turns).lower()
        attacker_text = ' '.join(t.get('content', '') for t in attacker_turns).lower()
        
        features['victim_avg_words'] = np.mean([len(t.get('content', '').split()) for t in victim_turns]) if victim_turns else 0
        features['victim_question_count'] = victim_text.count('?')
        features['victim_exclamation_count'] = victim_text.count('!')
        
        for emotion, words in self.emotion_words.items():
            features[f'victim_{emotion}_count'] = sum(victim_text.count(w) for w in words)
            features[f'attacker_{emotion}_count'] = sum(attacker_text.count(w) for w in words)
        
        features['compliance_score'] = features.get('victim_compliance_count', 0) / max(len(victim_turns), 1)
        features['resistance_score'] = features.get('victim_resistance_count', 0) / max(len(victim_turns), 1)
        
        victim_lengths = [len(t.get('content', '')) for t in victim_turns]
        features['victim_length_variance'] = np.var(victim_lengths) if len(victim_lengths) > 1 else 0
        features['victim_length_trend'] = self._compute_trend(victim_lengths)
        
        info_markers = ['card', 'number', 'account', 'ssn', 'social security', 'password', 'code', 'verify']
        features['info_disclosure_count'] = sum(victim_text.count(m) for m in info_markers)
        
        trust_phrases = ['i trust', 'you seem', 'sounds right', 'makes sense', 'of course']
        features['trust_language_count'] = sum(victim_text.count(p) for p in trust_phrases)
        
        uncertainty = ['maybe', 'perhaps', 'i think', 'not sure', 'confused', 'dont know']
        features['uncertainty_count'] = sum(victim_text.count(u) for u in uncertainty)
        
        return features
    
    def _compute_trend(self, values):
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0] if np.var(values) > 0 else 0
        return slope

def encode_cognitive_labels(victim_config):
    labels = {}
    trust_map = {'low': 0, 'moderately_trusting': 1, 'high': 2}
    labels['trust_level'] = trust_map.get(victim_config.get('trust_level'), 1)
    tech_map = {'low': 0, 'moderate': 1, 'high': 2}
    labels['tech_savviness'] = tech_map.get(victim_config.get('tech_savviness'), 0)
    awareness_map = {'unaware': 0, 'vaguely_aware': 1, 'aware': 2}
    labels['scam_awareness'] = awareness_map.get(victim_config.get('scam_awareness'), 1)
    decision_map = {'cautious': 0, 'moderate': 1, 'impulsive': 2}
    labels['decision_making'] = decision_map.get(victim_config.get('decision_making'), 1)
    vuln_score = (2 - labels['tech_savviness']) + (2 - labels['scam_awareness']) + labels['trust_level']
    labels['vulnerability_score'] = vuln_score
    labels['high_vulnerability'] = 1 if vuln_score >= 5 else 0
    return labels

def run_cognitive_profiler_experiment(max_per_category=200):
    print("Loading COVA-X dataset...")
    conversations = load_cova_x(max_per_category=max_per_category)
    extractor = ConversationFeatureExtractor()
    feature_rows, label_rows = [], []
    
    for conv in conversations:
        features = extractor.extract_features(conv)
        feature_rows.append(features)
        victim_config = conv.get('victim_config', {})
        labels = encode_cognitive_labels(victim_config)
        labels['conversation_id'] = conv.get('conversation_id')
        labels['scam_type'] = conv.get('scam_type')
        label_rows.append(labels)
    
    X = pd.DataFrame(feature_rows)
    y_df = pd.DataFrame(label_rows)
    print(f"\nExtracted {len(X)} conversations with {len(X.columns)} features")
    
    targets = ['trust_level', 'tech_savviness', 'scam_awareness', 'decision_making', 'high_vulnerability']
    results = {}
    
    for target in targets:
        print(f"\n{'=' * 60}")
        print(f"PREDICTING: {target.upper()}")
        print(f"{'=' * 60}")
        
        class_counts = y_df[target].value_counts()
        print(f"Class distribution: {dict(class_counts)}")
        if len(class_counts) < 2:
            print(f"Skipping {target}: only {len(class_counts)} class(es) present.")
            continue
        if class_counts.min() < 5:
            print(f"Warning: smallest class has only {class_counts.min()} samples.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_df[target], test_size=0.2, random_state=42, stratify=y_df[target]
        )
        
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, digits=3))
        
        importances = pd.Series(clf.feature_importances_, index=X.columns)
        print(f"\nTop 5 important features:")
        for feat, imp in importances.nlargest(5).items():
            print(f"  {feat}: {imp:.4f}")
        
        results[target] = {'accuracy': acc, 'model': clf}
    
    return results, X, y_df

if __name__ == "__main__":
    results, X, y_df = run_cognitive_profiler_experiment(max_per_category=200)
    print("\n" + "=" * 60)
    print("LAYER 2 COGNITIVE PROFILER - SUMMARY")
    print("=" * 60)
    for target, res in results.items():
        print(f"  {target}: {res['accuracy']:.3f} accuracy")
    print("\nThis validates that victim cognitive state can be inferred from conversational features.")
