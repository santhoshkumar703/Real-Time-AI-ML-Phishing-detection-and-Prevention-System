# Real-Time-AI-ML-Phishing-detection-and-Prevention-System
"""
SENTINEL AI - Real-Time Phishing Detection System
A comprehensive, production-ready architecture for AI-powered phishing prevention.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import base64
from urllib.parse import urlparse, parse_qs

# ML/Data Libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Conv1D, GlobalMaxPooling1D, Input, Concatenate, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# Image Processing (for visual similarity)
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim

# Web/Framework
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import redis
import kafka
from kafka import KafkaProducer, KafkaConsumer
import requests
import tldextract
import whois
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


# ============================================================================
# 1. DATA MODELS & ENUMS
# ============================================================================

class ThreatLevel(Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    PHISHING = "phishing"
    MALWARE = "malware"

class DetectionSource(Enum):
    URL_ANALYSIS = "url_analysis"
    EMAIL_CONTENT = "email_content"
    SENDER_REPUTATION = "sender_reputation"
    VISUAL_SIMILARITY = "visual_similarity"
    BEHAVIORAL = "behavioral"
    ENSEMBLE = "ensemble"

@dataclass
class DetectionResult:
    threat_level: ThreatLevel
    confidence: float
    source: DetectionSource
    features: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    explanation: str = ""
    recommended_action: str = ""

    def to_dict(self) -> Dict:
        return {
            "threat_level": self.threat_level.value,
            "confidence": round(self.confidence, 4),
            "source": self.source.value,
            "features": self.features,
            "timestamp": self.timestamp.isoformat(),
            "explanation": self.explanation,
            "recommended_action": self.recommended_action
        }

@dataclass
class URLFeatures:
    url: str
    domain_age_days: Optional[int] = None
    has_https: bool = False
    url_length: int = 0
    num_special_chars: int = 0
    num_subdomains: int = 0
    has_ip_address: bool = False
    has_suspicious_words: bool = False
    domain_reputation_score: float = 0.0
    alexa_rank: Optional[int] = None
    tld_risk_score: float = 0.0
    has_brand_name: bool = False
    brand_name: Optional[str] = None
    similarity_to_legitimate: float = 0.0

@dataclass
class EmailFeatures:
    sender_domain: str
    subject: str
    body_text: str
    has_attachments: bool = False
    num_links: int = 0
    spf_pass: bool = False
    dkim_pass: bool = False
    dmarc_pass: bool = False
    sender_reputation: float = 0.0
    urgency_score: float = 0.0
    grammar_score: float = 0.0
    sentiment_score: float = 0.0


# ============================================================================
# 2. FEATURE ENGINEERING MODULE
# ============================================================================

class FeatureEngineer:
    """Comprehensive feature extraction for phishing detection."""

    SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.top', '.xyz', '.buzz', '.click']
    SUSPICIOUS_KEYWORDS = ['urgent', 'verify', 'suspend', 'account', 'login', 'password',
                          'bank', 'update', 'security', 'confirm', 'limited', 'offer']
    BRAND_NAMES = ['amazon', 'apple', 'google', 'microsoft', 'facebook', 'paypal',
                   'netflix', 'bankofamerica', 'chase', 'wellsfargo']

    def __init__(self):
        self.tld_extractor = tldextract.TLDExtract()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def extract_url_features(self, url: str) -> URLFeatures:
        """Extract comprehensive URL-based features."""
        try:
            parsed = urlparse(url)
            extracted = self.tld_extractor(url)

            # Basic features
            has_https = parsed.scheme == 'https'
            url_length = len(url)
            num_special_chars = len(re.findall(r'[^a-zA-Z0-9]', url))
            num_subdomains = len(extracted.subdomain.split('.')) if extracted.subdomain else 0

            # Check for IP address
            has_ip = bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', extracted.domain))

            # Suspicious words
            domain_lower = (extracted.subdomain + extracted.domain).lower()
            has_suspicious = any(word in domain_lower for word in self.SUSPICIOUS_KEYWORDS)

            # TLD risk
            tld_risk = 0.8 if any(url.endswith(tld) for tld in self.SUSPICIOUS_TLDS) else 0.1

            # Brand impersonation detection
            has_brand, brand_name = self._detect_brand_impersonation(domain_lower)

            # Domain age (simulated - in production use WHOIS API)
            domain_age = self._get_domain_age(extracted.registered_domain)

            # Calculate similarity to legitimate domains (Levenshtein distance concept)
            similarity = self._calculate_similarity(domain_lower)

            return URLFeatures(
                url=url,
                domain_age_days=domain_age,
                has_https=has_https,
                url_length=url_length,
                num_special_chars=num_special_chars,
                num_subdomains=num_subdomains,
                has_ip_address=has_ip,
                has_suspicious_words=has_suspicious,
                tld_risk_score=tld_risk,
                has_brand_name=has_brand,
                brand_name=brand_name,
                similarity_to_legitimate=similarity
            )
        except Exception as e:
            logger.error(f"Error extracting URL features: {e}")
            return URLFeatures(url=url)

    def extract_email_features(self, subject: str, body: str, sender: str,
                              headers: Dict) -> EmailFeatures:
        """Extract email content and metadata features."""
        try:
            # Authentication results
            auth_results = headers.get('Authentication-Results', '')
            spf_pass = 'spf=pass' in auth_results
            dkim_pass = 'dkim=pass' in auth_results
            dmarc_pass = 'dmarc=pass' in auth_results

            # Content analysis
            combined_text = f"{subject} {body}".lower()
            urgency_words = ['urgent', 'immediate', 'alert', 'warning', 'suspend', 'verify now']
            urgency_score = sum(1 for word in urgency_words if word in combined_text) / len(urgency_words)

            # Grammar check (simplified - use language_tool in production)
            grammar_score = self._check_grammar(body)

            # Sentiment analysis
            sentiment = self.sentiment_analyzer.polarity_scores(body)
            sentiment_score = abs(sentiment['compound'])  # High absolute = emotional manipulation

            # Link extraction
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', body)

            return EmailFeatures(
                sender_domain=sender.split('@')[-1] if '@' in sender else '',
                subject=subject,
                body_text=body[:1000],  # Truncate for efficiency
                has_attachments='attachment' in headers.get('Content-Type', ''),
                num_links=len(urls),
                spf_pass=spf_pass,
                dkim_pass=dkim_pass,
                dmarc_pass=dmarc_pass,
                sender_reputation=self._get_sender_reputation(sender),
                urgency_score=min(urgency_score, 1.0),
                grammar_score=grammar_score,
                sentiment_score=sentiment_score
            )
        except Exception as e:
            logger.error(f"Error extracting email features: {e}")
            return EmailFeatures(sender_domain='', subject=subject, body_text=body[:1000])

    def _detect_brand_impersonation(self, domain: str) -> Tuple[bool, Optional[str]]:
        """Detect if domain is trying to impersonate a brand."""
        for brand in self.BRAND_NAMES:
            # Check for typosquatting (character substitution)
            if self._levenshtein_distance(domain, brand) <= 2 and domain != brand:
                return True, brand
            # Check for brand name inclusion with suspicious TLD
            if brand in domain and len(domain) > len(brand) + 3:
                return True, brand
        return False, None

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _calculate_similarity(self, domain: str) -> float:
        """Calculate similarity score to known legitimate domains."""
        max_similarity = 0.0
        for brand in self.BRAND_NAMES:
            dist = self._levenshtein_distance(domain, brand)
            similarity = 1 - (dist / max(len(domain), len(brand)))
            max_similarity = max(max_similarity, similarity)
        return max_similarity

    def _get_domain_age(self, domain: str) -> Optional[int]:
        """Get domain age in days (simulated - use WHOIS in production)."""
        # In production: use python-whois library
        # For demo, return random age weighted toward newer domains for phishing
        import random
        return random.randint(1, 3650)

    def _get_sender_reputation(self, sender: str) -> float:
        """Get sender reputation score (0-1, higher is better)."""
        # In production: query reputation databases (Abusix, Spamhaus, etc.)
        return 0.8

    def _check_grammar(self, text: str) -> float:
        """Check grammar quality (simplified)."""
        # In production: use language_tool_python or similar
        sentences = text.split('.')
        if not sentences:
            return 1.0
        # Simple heuristic: average sentence length
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        return 1.0 if 5 <= avg_len <= 20 else 0.5


# ============================================================================
# 3. ML MODELS IMPLEMENTATION
# ============================================================================

class BaseMLModel(ABC):
    """Abstract base class for ML models."""

    @abstractmethod
    def predict(self, features: Any) -> Tuple[ThreatLevel, float]:
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

class RandomForestPhishingDetector(BaseMLModel):
    """Random Forest model for structured feature detection."""

    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = []
        self.is_trained = False

    def _features_to_vector(self, features: URLFeatures) -> np.ndarray:
        """Convert URLFeatures to numerical vector."""
        return np.array([
            1 if features.has_https else 0,
            features.url_length,
            features.num_special_chars,
            features.num_subdomains,
            1 if features.has_ip_address else 0,
            1 if features.has_suspicious_words else 0,
            features.tld_risk_score,
            1 if features.has_brand_name else 0,
            features.similarity_to_legitimate,
            features.domain_age_days or 0,
            features.domain_reputation_score
        ]).reshape(1, -1)

    def predict(self, features: URLFeatures) -> Tuple[ThreatLevel, float]:
        if not self.is_trained:
            logger.warning("Model not trained, returning safe")
            return ThreatLevel.SAFE, 0.0

        X = self._features_to_vector(features)
        proba = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]

        confidence = max(proba)
        threat_map = {0: ThreatLevel.SAFE, 1: ThreatLevel.SUSPICIOUS, 2: ThreatLevel.PHISHING}
        return threat_map.get(pred, ThreatLevel.SAFE), confidence

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Random Forest model trained successfully")

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True

class NeuralNetworkPhishingDetector(BaseMLModel):
    """Deep Neural Network for complex pattern detection."""

    def __init__(self, input_dim: int = 11):
        self.input_dim = input_dim
        self.model = self._build_model()
        self.is_trained = False

    def _build_model(self) -> Model:
        """Build deep neural network architecture."""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes: Safe, Suspicious, Phishing
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def predict(self, features: URLFeatures) -> Tuple[ThreatLevel, float]:
        if not self.is_trained:
            return ThreatLevel.SAFE, 0.0

        X = self._features_to_vector(features)
        proba = self.model.predict(X, verbose=0)[0]
        pred = np.argmax(proba)

        confidence = float(max(proba))
        threat_map = {0: ThreatLevel.SAFE, 1: ThreatLevel.SUSPICIOUS, 2: ThreatLevel.PHISHING}
        return threat_map.get(pred, ThreatLevel.SAFE), confidence

    def _features_to_vector(self, features: URLFeatures) -> np.ndarray:
        """Convert features to normalized vector."""
        vector = np.array([
            1 if features.has_https else 0,
            min(features.url_length / 100, 1.0),  # Normalize
            min(features.num_special_chars / 20, 1.0),
            min(features.num_subdomains / 5, 1.0),
            1 if features.has_ip_address else 0,
            1 if features.has_suspicious_words else 0,
            features.tld_risk_score,
            1 if features.has_brand_name else 0,
            features.similarity_to_legitimate,
            min((features.domain_age_days or 0) / 3650, 1.0),
            features.domain_reputation_score
        ]).reshape(1, -1)
        return vector

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50) -> None:
        """Train neural network."""
        self.model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)
        self.is_trained = True
        logger.info("Neural Network trained successfully")

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = load_model(path)
        self.is_trained = True

class NLPPhishingDetector(BaseMLModel):
    """LSTM-based NLP model for text analysis."""

    def __init__(self, max_words: int = 10000, max_len: int = 200):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.model = None
        self.is_trained = False

    def _build_model(self) -> Model:
        """Build LSTM model for text classification."""
        model = Sequential([
            Embedding(self.max_words, 128, input_length=self.max_len),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def preprocess_text(self, texts: List[str]) -> np.ndarray:
        """Tokenize and pad sequences."""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        return padded

    def predict(self, text: str) -> Tuple[ThreatLevel, float]:
        if not self.is_trained:
            return ThreatLevel.SAFE, 0.0

        processed = self.preprocess_text([text])
        proba = self.model.predict(processed, verbose=0)[0]
        pred = np.argmax(proba)

        confidence = float(max(proba))
        threat_map = {0: ThreatLevel.SAFE, 1: ThreatLevel.SUSPICIOUS, 2: ThreatLevel.PHISHING}
        return threat_map.get(pred, ThreatLevel.SAFE), confidence

    def train(self, texts: List[str], labels: np.ndarray, epochs: int = 10) -> None:
        """Train NLP model."""
        self.tokenizer.fit_on_texts(texts)
        X = self.preprocess_text(texts)
        self.model = self._build_model()
        self.model.fit(X, labels, epochs=epochs, batch_size=64, validation_split=0.2, verbose=1)
        self.is_trained = True

    def save(self, path: str) -> None:
        self.model.save(f"{path}_model.h5")
        with open(f"{path}_tokenizer.pkl", 'wb') as f:
            pickle.dump(self.tokenizer, f)

    def load(self, path: str) -> None:
        self.model = load_model(f"{path}_model.h5")
        with open(f"{path}_tokenizer.pkl", 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.is_trained = True


# ============================================================================
# 4. DETECTION ENGINES
# ============================================================================

class URLAnalysisEngine:
    """Real-time URL analysis and reputation checking."""

    def __init__(self, rf_model: RandomForestPhishingDetector,
                 nn_model: NeuralNetworkPhishingDetector):
        self.feature_engineer = FeatureEngineer()
        self.rf_model = rf_model
        self.nn_model = nn_model
        self.cache = {}  # Simple LRU cache
        self.cache_lock = threading.Lock()

    def analyze(self, url: str, use_cache: bool = True) -> DetectionResult:
        """Analyze URL for phishing indicators."""
        # Check cache
        if use_cache:
            cache_key = hashlib.md5(url.encode()).hexdigest()
            with self.cache_lock:
                if cache_key in self.cache:
                    cached = self.cache[cache_key]
                    if (datetime.now() - cached['timestamp']).seconds < 300:  # 5 min TTL
                        return cached['result']

        # Extract features
        features = self.feature_engineer.extract_url_features(url)

        # Get predictions from both models
        rf_threat, rf_conf = self.rf_model.predict(features)
        nn_threat, nn_conf = self.nn_model.predict(features)

        # Ensemble decision (weighted average)
        # Weight NN higher for complex patterns, RF for interpretability
        threat_scores = {
            ThreatLevel.SAFE: 0,
            ThreatLevel.SUSPICIOUS: 0,
            ThreatLevel.PHISHING: 0
        }

        threat_scores[rf_threat] += rf_conf * 0.4
        threat_scores[nn_threat] += nn_conf * 0.6

        final_threat = max(threat_scores, key=threat_scores.get)
        final_conf = threat_scores[final_threat]

        # Generate explanation
        explanation = self._generate_explanation(features, final_threat)

        result = DetectionResult(
            threat_level=final_threat,
            confidence=final_conf,
            source=DetectionSource.URL_ANALYSIS,
            features=features.__dict__,
            explanation=explanation,
            recommended_action=self._get_recommendation(final_threat)
        )

        # Update cache
        if use_cache:
            with self.cache_lock:
                self.cache[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now()
                }

        return result

    def _generate_explanation(self, features: URLFeatures, threat: ThreatLevel) -> str:
        """Generate human-readable explanation."""
        reasons = []
        if features.has_brand_name:
            reasons.append(f"Brand impersonation detected: {features.brand_name}")
        if features.tld_risk_score > 0.5:
            reasons.append("High-risk top-level domain")
        if not features.has_https:
            reasons.append("No HTTPS encryption")
        if features.similarity_to_legitimate > 0.8:
            reasons.append("High similarity to legitimate domain (possible typosquatting)")
        if features.domain_age_days and features.domain_age_days < 30:
            reasons.append("Recently registered domain")

        return " | ".join(reasons) if reasons else "No specific threats detected"

    def _get_recommendation(self, threat: ThreatLevel) -> str:
        """Get recommended action based on threat level."""
        actions = {
            ThreatLevel.SAFE: "Allow connection",
            ThreatLevel.SUSPICIOUS: "Flag for review, allow with monitoring",
            ThreatLevel.PHISHING: "Block immediately, alert security team"
        }
        return actions.get(threat, "Review manually")

class EmailAnalysisEngine:
    """Email content and metadata analysis."""

    def __init__(self, nlp_model: NLPPhishingDetector):
        self.feature_engineer = FeatureEngineer()
        self.nlp_model = nlp_model

    def analyze(self, subject: str, body: str, sender: str,
                headers: Dict) -> DetectionResult:
        """Analyze email for phishing indicators."""
        features = self.feature_engineer.extract_email_features(subject, body, sender, headers)

        # Combine subject and body for NLP analysis
        full_text = f"{subject} {body}"
        nlp_threat, nlp_conf = self.nlp_model.predict(full_text)

        # Rule-based checks
        rule_score = 0
        if not features.spf_pass:
            rule_score += 0.3
        if not features.dkim_pass:
            rule_score += 0.2
        if not features.dmarc_pass:
            rule_score += 0.2
        if features.urgency_score > 0.5:
            rule_score += 0.2
        if features.sender_reputation < 0.5:
            rule_score += 0.3

        # Combine NLP and rule-based
        final_conf = (nlp_conf * 0.7) + (rule_score * 0.3)

        if final_conf > 0.8:
            final_threat = ThreatLevel.PHISHING
        elif final_conf > 0.5:
            final_threat = ThreatLevel.SUSPICIOUS
        else:
            final_threat = ThreatLevel.SAFE

        explanation = self._generate_email_explanation(features, final_threat)

        return DetectionResult(
            threat_level=final_threat,
            confidence=final_conf,
            source=DetectionSource.EMAIL_CONTENT,
            features=features.__dict__,
            explanation=explanation,
            recommended_action=self._get_email_recommendation(final_threat)
        )

    def _generate_email_explanation(self, features: EmailFeatures, threat: ThreatLevel) -> str:
        reasons = []
        if not features.spf_pass:
            reasons.append("SPF authentication failed")
        if features.urgency_score > 0.5:
            reasons.append("Urgency tactics detected")
        if features.num_links > 5:
            reasons.append("Multiple suspicious links")
        if features.has_attachments:
            reasons.append("Contains attachments")
        return " | ".join(reasons) if reasons else "Standard email patterns"

    def _get_email_recommendation(self, threat: ThreatLevel) -> str:
        actions = {
            ThreatLevel.SAFE: "Deliver to inbox",
            ThreatLevel.SUSPICIOUS: "Move to spam folder",
            ThreatLevel.PHISHING: "Quarantine and notify admin"
        }
        return actions.get(threat, "Manual review")

class VisualSimilarityEngine:
    """Detect visual phishing via screenshot comparison."""

    def __init__(self, brand_logos_path: str = "./brand_logos"):
        self.brand_logos = {}
        self.known_brands = ['amazon', 'paypal', 'apple', 'microsoft', 'google']
        self._load_brand_logos(brand_logos_path)

    def _load_brand_logos(self, path: str):
        """Load reference brand logos."""
        # In production, load actual image files
        # For demo, we'll simulate with hashes
        for brand in self.known_brands:
            self.brand_logos[brand] = f"hash_{brand}"

    def analyze_screenshot(self, image_path: str, claimed_brand: Optional[str] = None) -> DetectionResult:
        """
        Analyze webpage screenshot for visual similarity to brand logos.
        Uses SSIM (Structural Similarity Index) and perceptual hashing.
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                # Create dummy image for demo
                img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            max_similarity = 0.0
            detected_brand = None

            # Compare with known brands (simulated)
            for brand in self.known_brands:
                # In production: compare with actual logo templates
                # using template matching or deep learning (Siamese networks)
                similarity = self._calculate_visual_similarity(gray, brand)
                if similarity > max_similarity:
                    max_similarity = similarity
                    detected_brand = brand

            # Determine threat level
            if max_similarity > 0.85 and claimed_brand != detected_brand:
                threat = ThreatLevel.PHISHING
                conf = max_similarity
                explanation = f"Visual impersonation of {detected_brand} detected"
            elif max_similarity > 0.7:
                threat = ThreatLevel.SUSPICIOUS
                conf = max_similarity
                explanation = f"High visual similarity to {detected_brand}"
            else:
                threat = ThreatLevel.SAFE
                conf = 1 - max_similarity
                explanation = "No brand impersonation detected"

            return DetectionResult(
                threat_level=threat,
                confidence=conf,
                source=DetectionSource.VISUAL_SIMILARITY,
                features={
                    "detected_brand": detected_brand,
                    "similarity_score": max_similarity,
                    "claimed_brand": claimed_brand
                },
                explanation=explanation,
                recommended_action="Block if impersonating financial brand" if threat == ThreatLevel.PHISHING else "Allow"
            )

        except Exception as e:
            logger.error(f"Visual analysis error: {e}")
            return DetectionResult(
                threat_level=ThreatLevel.SAFE,
                confidence=0.0,
                source=DetectionSource.VISUAL_SIMILARITY,
                features={},
                explanation="Analysis failed",
                recommended_action="Manual review"
            )

    def _calculate_visual_similarity(self, img: np.ndarray, brand: str) -> float:
        """Calculate similarity score (simulated)."""
        # In production: Use SSIM, perceptual hashing, or CNN features
        import random
        return random.uniform(0.3, 0.95)


# ============================================================================
# 5. REAL-TIME PROCESSING & STREAMING
# ============================================================================

class StreamProcessor:
    """
    High-performance stream processing using asyncio and batching.
    Optimized for sub-100ms latency at scale.
    """

    def __init__(self, url_engine: URLAnalysisEngine,
                 email_engine: EmailAnalysisEngine,
                 visual_engine: VisualSimilarityEngine,
                 batch_size: int = 32,
                 max_latency_ms: float = 50.0):
        self.url_engine = url_engine
        self.email_engine = email_engine
        self.visual_engine = visual_engine
        self.batch_size = batch_size
        self.max_latency_ms = max_latency_ms

        # Async processing queues
        self.url_queue = asyncio.Queue(maxsize=1000)
        self.email_queue = asyncio.Queue(maxsize=1000)
        self.visual_queue = asyncio.Queue(maxsize=100)

        # Metrics
        self.processed_count = 0
        self.latency_history = deque(maxlen=1000)
        self.error_count = 0

        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=8)

    async def start(self):
        """Start stream processing workers."""
        await asyncio.gather(
            self._url_worker(),
            self._email_worker(),
            self._visual_worker(),
            self._metrics_reporter()
        )

    async def submit_url(self, url: str) -> asyncio.Future:
        """Submit URL for analysis."""
        future = asyncio.get_event_loop().create_future()
        await self.url_queue.put((url, future))
        return future

    async def submit_email(self, email_data: Dict) -> asyncio.Future:
        """Submit email for analysis."""
        future = asyncio.get_event_loop().create_future()
        await self.email_queue.put((email_data, future))
        return future

    async def _url_worker(self):
        """Process URL batch with dynamic batching."""
        batch = []
        futures = []
        last_process_time = time.time()

        while True:
            try:
                # Wait for item with timeout for dynamic batching
                timeout = self.max_latency_ms / 1000.0
                url, future = await asyncio.wait_for(
                    self.url_queue.get(),
                    timeout=max(0, timeout - (time.time() - last_process_time))
                )
                batch.append(url)
                futures.append(future)

                # Process batch if full or timeout reached
                if len(batch) >= self.batch_size or \
                   (time.time() - last_process_time) * 1000 >= self.max_latency_ms:
                    await self._process_url_batch(batch, futures)
                    batch = []
                    futures = []
                    last_process_time = time.time()

            except asyncio.TimeoutError:
                if batch:
                    await self._process_url_batch(batch, futures)
                    batch = []
                    futures = []
                    last_process_time = time.time()

    async def _process_url_batch(self, urls: List[str], futures: List[asyncio.Future]):
        """Process batch of URLs concurrently."""
        start_time = time.time()

        # Run CPU-intensive ML inference in thread pool
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self.url_engine.analyze, url)
            for url in urls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Resolve futures with results
        for future, result in zip(futures, results):
            if isinstance(result, Exception):
                self.error_count += 1
                future.set_exception(result)
            else:
                future.set_result(result)
                self.processed_count += 1

        # Record latency
        latency = (time.time() - start_time) * 1000 / len(urls)  # per item
        self.latency_history.append(latency)

    async def _email_worker(self):
        """Process email stream."""
        while True:
            email_data, future = await self.email_queue.get()
            try:
                start_time = time.time()
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.email_engine.analyze,
                    email_data['subject'],
                    email_data['body'],
                    email_data['sender'],
                    email_data.get('headers', {})
                )
                future.set_result(result)
                self.latency_history.append((time.time() - start_time) * 1000)
            except Exception as e:
                self.error_count += 1
                future.set_exception(e)

    async def _visual_worker(self):
        """Process visual analysis (lower throughput)."""
        while True:
            visual_data, future = await self.visual_queue.get()
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.visual_engine.analyze_screenshot,
                    visual_data['image_path'],
                    visual_data.get('claimed_brand')
                )
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    async def _metrics_reporter(self):
        """Report performance metrics periodically."""
        while True:
            await asyncio.sleep(60)  # Every minute
            if self.latency_history:
                avg_latency = sum(self.latency_history) / len(self.latency_history)
                p95_latency = sorted(self.latency_history)[int(len(self.latency_history) * 0.95)]
                logger.info(f"Stream Metrics - Processed: {self.processed_count}, "
                          f"Avg Latency: {avg_latency:.2f}ms, "
                          f"P95: {p95_latency:.2f}ms, "
                          f"Errors: {self.error_count}")


# ============================================================================
# 6. ENSEMBLE ORCHESTRATOR
# ============================================================================

class EnsembleOrchestrator:
    """
    Advanced ensemble system combining multiple detection engines
    with weighted voting and confidence calibration.
    """

    def __init__(self):
        self.engines = {}
        self.weights = {
            DetectionSource.URL_ANALYSIS: 0.35,
            DetectionSource.EMAIL_CONTENT: 0.30,
            DetectionSource.VISUAL_SIMILARITY: 0.20,
            DetectionSource.SENDER_REPUTATION: 0.15
        }
        self.history = deque(maxlen=10000)  # For online learning

    def register_engine(self, name: str, engine: Any, weight: float):
        """Register a detection engine."""
        self.engines[name] = engine
        self.weights[name] = weight

    async def analyze(self, data: Dict) -> DetectionResult:
        """
        Run ensemble analysis across all applicable engines.

        Args:
            data: Dictionary containing 'url', 'email', 'image', etc.
        """
        results = []

        # Parallel execution of all applicable engines
        tasks = []

        if 'url' in data:
            tasks.append(self._run_engine('url', data['url']))
        if 'email' in data:
            tasks.append(self._run_engine('email', data['email']))
        if 'image' in data:
            tasks.append(self._run_engine('visual', data['image']))

        engine_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_results = [r for r in engine_results if not isinstance(r, Exception)]

        if not valid_results:
            return DetectionResult(
                threat_level=ThreatLevel.SAFE,
                confidence=0.0,
                source=DetectionSource.ENSEMBLE,
                features={},
                explanation="All engines failed",
                recommended_action="Manual review"
            )

        # Weighted voting
        threat_votes = {level: 0.0 for level in ThreatLevel}
        total_confidence = 0.0

        for result in valid_results:
            weight = self.weights.get(result.source, 0.25)
            threat_votes[result.threat_level] += result.confidence * weight
            total_confidence += result.confidence * weight

        # Normalize
        if total_confidence > 0:
            for level in threat_votes:
                threat_votes[level] /= total_confidence

        # Select winner
        final_threat = max(threat_votes, key=threat_votes.get)
        final_conf = threat_votes[final_threat]

        # Meta-features for ensemble decision
        meta_features = {
            "engine_results": [r.to_dict() for r in valid_results],
            "vote_distribution": {k.value: v for k, v in threat_votes.items()},
            "engines_used": len(valid_results)
        }

        # Adaptive weight adjustment (online learning)
        self._update_weights(valid_results, final_threat)

        result = DetectionResult(
            threat_level=final_threat,
            confidence=final_conf,
            source=DetectionSource.ENSEMBLE,
            features=meta_features,
            explanation=f"Ensemble decision based on {len(valid_results)} engines",
            recommended_action=self._get_ensemble_recommendation(final_threat, final_conf)
        )

        self.history.append((data, result))
        return result

    async def _run_engine(self, engine_type: str, data: Any) -> DetectionResult:
        """Run specific engine."""
        if engine_type == 'url':
            return await asyncio.get_event_loop().run_in_executor(
                None, self.engines['url'].analyze, data
            )
        elif engine_type == 'email':
            return await asyncio.get_event_loop().run_in_executor(
                None, self.engines['email'].analyze,
                data['subject'], data['body'], data['sender'], data.get('headers', {})
            )
        elif engine_type == 'visual':
            return await asyncio.get_event_loop().run_in_executor(
                None, self.engines['visual'].analyze_screenshot,
                data['path'], data.get('brand')
            )

    def _update_weights(self, results: List[DetectionResult], final_decision: ThreatLevel):
        """Simple online learning - boost weights of engines agreeing with consensus."""
        # In production: use more sophisticated methods (multi-armed bandit, etc.)
        pass

    def _get_ensemble_recommendation(self, threat: ThreatLevel, conf: float) -> str:
        if threat == ThreatLevel.PHISHING and conf > 0.9:
            return "Immediate block and SOC alert"
        elif threat == ThreatLevel.PHISHING:
            return "Block and log"
        elif threat == ThreatLevel.SUSPICIOUS:
            return "Flag for review"
        return "Allow"


# ============================================================================
# 7. API & DEPLOYMENT
# ============================================================================

# Pydantic models for API
class URLScanRequest(BaseModel):
    url: str = Field(..., description="URL to analyze")
    client_id: Optional[str] = Field(None, description="Client identifier")
    metadata: Optional[Dict] = Field(default_factory=dict)

class EmailScanRequest(BaseModel):
    subject: str
    body: str
    sender: str
    headers: Optional[Dict] = Field(default_factory=dict)
    client_id: Optional[str] = None

class ScanResponse(BaseModel):
    threat_level: str
    confidence: float
    source: str
    explanation: str
    recommended_action: str
    timestamp: str
    request_id: str
    processing_time_ms: float

class PhishingDetectionAPI:
    """FastAPI-based production API with WebSocket support."""

    def __init__(self):
        self.app = FastAPI(
            title="Sentinel AI - Phishing Detection API",
            description="Real-time ML-powered phishing detection",
            version="2.0.0"
        )
        self.setup_middleware()
        self.setup_routes()

        # Initialize components
        self._init_models()

    def _init_models(self):
        """Initialize ML models."""
        # Initialize models (in production, load pre-trained weights)
        self.rf_model = RandomForestPhishingDetector()
        self.nn_model = NeuralNetworkPhishingDetector()
        self.nlp_model = NLPPhishingDetector()

        # Create dummy training data for demo
        self._train_dummy_models()

        # Initialize engines
        self.url_engine = URLAnalysisEngine(self.rf_model, self.nn_model)
        self.email_engine = EmailAnalysisEngine(self.nlp_model)
        self.visual_engine = VisualSimilarityEngine()

        # Initialize stream processor
        self.stream_processor = StreamProcessor(
            self.url_engine, self.email_engine, self.visual_engine
        )

        # Initialize orchestrator
        self.orchestrator = EnsembleOrchestrator()
        self.orchestrator.register_engine('url', self.url_engine, 0.35)
        self.orchestrator.register_engine('email', self.email_engine, 0.30)
        self.orchestrator.register_engine('visual', self.visual_engine, 0.20)

        # Start background processing

    import asyncio

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(self.stream_processor.start())
    except RuntimeError:
        pass

    def _train_dummy_models(self):
        """Create dummy trained models for demonstration."""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000

        # URL features: [https, length, special_chars, subdomains, ip, suspicious_words, tld_risk, brand, similarity, age, reputation]
        X_url = np.random.rand(n_samples, 11)
        y_url = np.random.randint(0, 3, n_samples)  # 0: safe, 1: suspicious, 2: phishing

        # Make phishing samples more realistic
        phishing_idx = np.where(y_url == 2)[0]
        X_url[phishing_idx, 0] = 0  # No HTTPS
        X_url[phishing_idx, 4] = 1  # Has IP
        X_url[phishing_idx, 6] = 0.8  # High TLD risk

        self.rf_model.train(X_url, y_url)
        self.nn_model.train(X_url, y_url, epochs=5)

        # NLP model
        texts = [
            "Verify your account immediately" if i % 3 == 0 else
            "Meeting reminder for tomorrow" if i % 3 == 1 else
            "Your package has been shipped"
            for i in range(n_samples)
        ]
        y_text = np.array([2 if i % 3 == 0 else 0 for i in range(n_samples)])
        self.nlp_model.train(texts, y_text, epochs=2)

        logger.info("Dummy models trained successfully")

    def setup_middleware(self):
        """Setup CORS and middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        """Setup API routes."""

        @self.app.post("/scan/url", response_model=ScanResponse)
        async def scan_url(request: URLScanRequest, background_tasks: BackgroundTasks):
            """Scan a URL for phishing indicators."""
            start_time = time.time()
            request_id = hashlib.md5(f"{request.url}{time.time()}".encode()).hexdigest()[:12]

            try:
                # Use stream processor for async handling
                future = await self.stream_processor.submit_url(request.url)
                result = await asyncio.wait_for(future, timeout=5.0)

                processing_time = (time.time() - start_time) * 1000

                return ScanResponse(
                    threat_level=result.threat_level.value,
                    confidence=result.confidence,
                    source=result.source.value,
                    explanation=result.explanation,
                    recommended_action=result.recommended_action,
                    timestamp=result.timestamp.isoformat(),
                    request_id=request_id,
                    processing_time_ms=processing_time
                )

            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Analysis timeout")
            except Exception as e:
                logger.error(f"URL scan error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/scan/email", response_model=ScanResponse)
        async def scan_email(request: EmailScanRequest):
            """Scan an email for phishing indicators."""
            start_time = time.time()
            request_id = hashlib.md5(f"{request.sender}{time.time()}".encode()).hexdigest()[:12]

            try:
                future = await self.stream_processor.submit_email({
                    'subject': request.subject,
                    'body': request.body,
                    'sender': request.sender,
                    'headers': request.headers
                })
                result = await asyncio.wait_for(future, timeout=5.0)

                processing_time = (time.time() - start_time) * 1000

                return ScanResponse(
                    threat_level=result.threat_level.value,
                    confidence=result.confidence,
                    source=result.source.value,
                    explanation=result.explanation,
                    recommended_action=result.recommended_action,
                    timestamp=result.timestamp.isoformat(),
                    request_id=request_id,
                    processing_time_ms=processing_time
                )

            except Exception as e:
                logger.error(f"Email scan error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/scan/ensemble")
        async def scan_ensemble(data: Dict):
            """Run ensemble analysis on multiple data sources."""
            result = await self.orchestrator.analyze(data)
            return result.to_dict()

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "models_loaded": True,
                "queue_depth": self.stream_processor.url_queue.qsize()
            }

        @self.app.get("/metrics")
        async def get_metrics():
            """Get system metrics."""
            return {
                "processed_count": self.stream_processor.processed_count,
                "error_count": self.stream_processor.error_count,
                "avg_latency_ms": sum(self.stream_processor.latency_history) / len(self.stream_processor.latency_history) if self.stream_processor.latency_history else 0,
                "p95_latency_ms": sorted(self.stream_processor.latency_history)[int(len(self.stream_processor.latency_history) * 0.95)] if len(self.stream_processor.latency_history) > 20 else 0
            }

        @self.app.websocket("/ws/stream")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time streaming updates."""
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_json()
                    result = await self.orchestrator.analyze(data)
                    await websocket.send_json(result.to_dict())
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.close()

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server."""
        uvicorn.run(self.app, host=host, port=port)


# ============================================================================
# 8. EVALUATION & BENCHMARKING
# ============================================================================

class ModelEvaluator:
    """Comprehensive evaluation framework."""

    def __init__(self):
        self.results = []

    def evaluate_model(self, model: BaseMLModel, X_test: np.ndarray,
                       y_test: np.ndarray, model_name: str) -> Dict:
        """Evaluate single model performance."""
        predictions = []
        probabilities = []

        # Handle different input types
        for i in range(len(X_test)):
            if isinstance(model, (RandomForestPhishingDetector, NeuralNetworkPhishingDetector)):
                # Convert array back to URLFeatures for demo
                features = URLFeatures(url=f"test_{i}")
                pred, conf = model.predict(features)
                predictions.append(2 if pred == ThreatLevel.PHISHING else
                                 1 if pred == ThreatLevel.SUSPICIOUS else 0)
                probabilities.append(conf)
            else:
                pred, conf = model.predict("test text")
                predictions.append(2 if pred == ThreatLevel.PHISHING else
                                 1 if pred == ThreatLevel.SUSPICIOUS else 0)
                probabilities.append(conf)

        predictions = np.array(predictions)

        # Calculate metrics
        metrics = {
            "model": model_name,
            "precision_macro": precision_score(y_test, predictions, average='macro', zero_division=0),
            "recall_macro": recall_score(y_test, predictions, average='macro', zero_division=0),
            "f1_macro": f1_score(y_test, predictions, average='macro', zero_division=0),
            "precision_phishing": precision_score(y_test == 2, predictions == 2, zero_division=0),
            "recall_phishing": recall_score(y_test == 2, predictions == 2, zero_division=0),
            "false_positive_rate": np.sum((predictions == 2) & (y_test != 2)) / np.sum(y_test != 2),
            "false_negative_rate": np.sum((predictions != 2) & (y_test == 2)) / np.sum(y_test == 2),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist()
        }

        self.results.append(metrics)
        return metrics

    def benchmark_latency(self, model: BaseMLModel, n_samples: int = 1000) -> Dict:
        """Benchmark inference latency."""
        latencies = []

        for _ in range(n_samples):
            start = time.perf_counter()
            if isinstance(model, (RandomForestPhishingDetector, NeuralNetworkPhishingDetector)):
                features = URLFeatures(url="http://example.com")
                model.predict(features)
            else:
                model.predict("test email content")
            latencies.append((time.perf_counter() - start) * 1000)  # ms

        return {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies)
        }

    def generate_report(self) -> str:
        """Generate evaluation report."""
        report = ["# Model Evaluation Report\n"]

        for result in self.results:
            report.append(f"\n## {result['model']}")
            report.append(f"- Precision (Phishing): {result['precision_phishing']:.4f}")
            report.append(f"- Recall (Phishing): {result['recall_phishing']:.4f}")
            report.append(f"- F1 Score (Macro): {result['f1_macro']:.4f}")
            report.append(f"- False Positive Rate: {result['false_positive_rate']:.4f}")
            report.append(f"- False Negative Rate: {result['false_negative_rate']:.4f}")

        return "\n".join(report)


# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           SENTINEL AI - Phishing Detection System            ║
    ║                                                              ║
    ║  Real-time ML-powered threat detection and prevention       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Initialize and run API
    import asyncio

    async def main():
        api = PhishingDetectionAPI()
        api.run()

    if __name__ == "__main__":
        main()

    print("Starting API server on http://0.0.0.0:8000")
    print("Documentation available at: http://0.0.0.0:8000/docs")
    print("\nTest commands:")
    print("  curl -X POST http://localhost:8000/scan/url \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"url\": \"http://suspicious-bank-login.com\"}'")


def main():
    api = PhishingDetectionAPI()
    api.run()

if __name__ == "__main__":
    main()
