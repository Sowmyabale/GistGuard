"""
ui_demo.py
Simple Flask web demo for GXAI-Elderly real-time intervention.
Shows how the system would look when deployed as a protective agent.

Run locally:
    pip install flask
    python ui_demo.py
    Open http://localhost:5000 in browser

Features:
    - Simulated incoming scam text
    - Layer 1: Scam detection with confidence scores
    - Layer 2: Cognitive profile inference
    - Layer 3: Gist explanation generation
    - User feedback: thumbs up/down
"""

from flask import Flask, render_template_string, request, jsonify
import os
import sys
import json
import random

app = Flask(__name__)

# ============================================================================
# SIMULATED DATA (replace with actual model inference in production)
# ============================================================================
SAMPLE_SCAMS = [
    {
        "sender": "+1-555-0199",
        "text": "URGENT: Your bank account has been compromised. Verify your details immediately at secure-bank-update.com or your funds will be frozen within 2 hours. Do not ignore this message.",
        "scam_type": "bank",
        "tactics": ["urgency", "fear_appeal", "false_urgency"]
    },
    {
        "sender": "IRS-Department",
        "text": "This is the IRS. You owe $4,327 in back taxes. A warrant for your arrest has been issued. Call 1-800-SCAM-NOW immediately to resolve this before officers arrive at your home.",
        "scam_type": "government_impersonation",
        "tactics": ["authority_claim", "fear_appeal", "urgency"]
    },
    {
        "sender": "Grandson Jake",
        "text": "Grandma it's Jake I'm in jail in Mexico please don't tell mom I need $2000 for bail send it to this Western Union right away I'm so scared",
        "scam_type": "grandparent",
        "tactics": ["urgency", "social_proof"]
    },
    {
        "sender": "Jessica",
        "text": "Hi! We matched on that dating site last week. I've been thinking about you. I'm actually a crypto trader and I've made so much money. Want me to show you how? It's guaranteed returns.",
        "scam_type": "romance",
        "tactics": ["social_proof", "false_urgency"]
    },
]

def simulate_layer1(text):
    """Simulate Layer 1: Scam detection."""
    # In production: load BERT model and predict
    scam = random.choice(SAMPLE_SCAMS)
    return {
        "scam_type": scam["scam_type"],
        "confidence": round(random.uniform(0.85, 0.99), 3),
        "is_scam": True,
        "tactics": scam["tactics"]
    }

def simulate_layer2(text):
    """Simulate Layer 2: Cognitive profiling."""
    # In production: load BiLSTM profiler
    return {
        "trust_level": random.choice(["low", "moderate", "high"]),
        "tech_savviness": random.choice(["low", "moderate", "high"]),
        "scam_awareness": random.choice(["unaware", "vaguely_aware", "aware"])
    }

def simulate_layer3(scam_type, profile):
    """Simulate Layer 3: Gist generation."""
    # In production: load LoRA LLM and generate
    gists = {
        "bank": {
            "low": "This message is pretending to be from your bank to steal your information. Real banks never send links in texts. Do not click anything. Call the number on your bank card instead.",
            "high": "This is a smishing attack using urgency and a fake bank domain. Do not click the link. Verify directly through your bank's official app or website."
        },
        "government_impersonation": {
            "low": "The government does not text people about arrest warrants. This is a scam to scare you. Hang up and do not call the number they gave you.",
            "high": "This is a classic IRS impersonation scam. The IRS never demands immediate payment via phone/text or threatens arrest. Report to treasury.gov."
        },
        "grandparent": {
            "low": "This is not your grandchild. Scammers pretend to be family in emergencies to get money. Call your grandchild directly on their real number to check.",
            "high": "This is a virtual kidnapping/grandparent scam. The 'emergency' is fabricated. Verify independently before sending any money."
        },
        "romance": {
            "low": "This person is not who they say they are. They will ask for money soon. Do not send money or share personal information with anyone you have not met in person.",
            "high": "This is a romance scam (pig butchering variant). The 'guaranteed returns' claim is a red flag. Never invest based on online relationships."
        }
    }
    
    tech = profile["tech_savviness"]
    gist = gists.get(scam_type, {}).get(tech, gists["bank"]["low"])
    return gist

# ============================================================================
# FLASK ROUTES
# ============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>GXAI-Elderly Protective Agent</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .header { background: #2D5F4A; color: white; padding: 20px; border-radius: 10px; text-align: center; }
        .phone-screen { background: white; border: 2px solid #ccc; border-radius: 20px; padding: 20px; margin: 20px 0; }
        .message { background: #e3f2fd; padding: 15px; border-radius: 15px; margin: 10px 0; }
        .alert { background: #ffebee; border-left: 4px solid #f44336; padding: 15px; margin: 15px 0; }
        .gist { background: #e8f5e9; border-left: 4px solid #4caf50; padding: 15px; margin: 15px 0; }
        .profile { background: #fff3e0; padding: 15px; border-radius: 10px; margin: 10px 0; }
        .button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .btn-danger { background: #f44336; color: white; }
        .btn-safe { background: #4caf50; color: white; }
        .btn-action { background: #2196f3; color: white; }
        .hidden { display: none; }
        .stats { display: flex; justify-content: space-around; margin: 15px 0; }
        .stat-box { text-align: center; padding: 10px; background: white; border-radius: 10px; flex: 1; margin: 0 5px; }
        .stat-number { font-size: 24px; font-weight: bold; color: #2D5F4A; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ GXAI-Elderly Protective Agent</h1>
        <p>Real-time scam detection and gist-based intervention</p>
    </div>
    
    <div class="stats">
        <div class="stat-box">
            <div class="stat-number">{{ stats.scams_blocked }}</div>
            <div>Scams Blocked</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{{ stats.active_users }}</div>
            <div>Active Users</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{{ stats.accuracy }}%</div>
            <div>Detection Accuracy</div>
        </div>
    </div>
    
    <div class="phone-screen">
        <h3>📱 Simulated Incoming Message</h3>
        <div class="message">
            <strong>From:</strong> {{ message.sender }}<br>
            <strong>Message:</strong> {{ message.text }}
        </div>
        
        <button class="button btn-action" onclick="analyzeMessage()">🔍 Analyze Message</button>
        <button class="button btn-action" onclick="loadNewMessage()">🔄 New Message</button>
    </div>
    
    <div id="analysis-result" class="hidden">
        <div class="alert">
            <h3>⚠️ SCAM DETECTED</h3>
            <p><strong>Type:</strong> <span id="scam-type"></span></p>
            <p><strong>Confidence:</strong> <span id="confidence"></span></p>
            <p><strong>Manipulation Tactics:</strong> <span id="tactics"></span></p>
        </div>
        
        <div class="profile">
            <h3>🧠 Inferred Cognitive Profile</h3>
            <p>Trust Level: <span id="trust"></span></p>
            <p>Tech Savviness: <span id="tech"></span></p>
            <p>Scam Awareness: <span id="awareness"></span></p>
        </div>
        
        <div class="gist">
            <h3>💡 Generated Gist Explanation</h3>
            <p id="gist-text"></p>
        </div>
        
        <div style="text-align: center; margin-top: 20px;">
            <p>Was this explanation helpful?</p>
            <button class="button btn-safe" onclick="submitFeedback(true)">👍 Yes</button>
            <button class="button btn-danger" onclick="submitFeedback(false)">👎 No</button>
        </div>
    </div>
    
    <div id="feedback-result" class="hidden" style="text-align: center; padding: 20px;">
        <h3>Thank you for your feedback!</h3>
        <p>The system will use this to improve future explanations.</p>
    </div>
    
    <script>
        function analyzeMessage() {
            fetch('/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: `{{ message.text|replace('\\n', ' ') }}`, sender: '{{ message.sender }}'})
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('scam-type').textContent = data.scam_type;
                document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
                document.getElementById('tactics').textContent = data.tactics.join(', ');
                document.getElementById('trust').textContent = data.profile.trust_level;
                document.getElementById('tech').textContent = data.profile.tech_savviness;
                document.getElementById('awareness').textContent = data.profile.scam_awareness;
                document.getElementById('gist-text').textContent = data.gist;
                document.getElementById('analysis-result').classList.remove('hidden');
            });
        }
        
        function loadNewMessage() {
            location.reload();
        }
        
        function submitFeedback(helpful) {
            fetch('/feedback', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({helpful: helpful})
            });
            document.getElementById('feedback-result').classList.remove('hidden');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    message = random.choice(SAMPLE_SCAMS)
    stats = {
        "scams_blocked": 1247,
        "active_users": 89,
        "accuracy": 99.6
    }
    return render_template_string(HTML_TEMPLATE, message=message, stats=stats)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    
    # Run simulated pipeline
    layer1_result = simulate_layer1(text)
    layer2_result = simulate_layer2(text)
    gist = simulate_layer3(layer1_result['scam_type'], layer2_result)
    
    return jsonify({
        'scam_type': layer1_result['scam_type'],
        'confidence': layer1_result['confidence'],
        'tactics': layer1_result['tactics'],
        'profile': layer2_result,
        'gist': gist
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    helpful = data.get('helpful', False)
    # In production: store feedback, trigger RL update
    print(f"Feedback received: helpful={helpful}")
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    print("=" * 60)
    print("GXAI-Elderly UI Demo")
    print("=" * 60)
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)