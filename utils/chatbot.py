"""
BrahamAI — Smart Rule-Based Chatbot
Provides travel suggestions via multi-keyword matching and budget detection.

Improvements over v1:
  - Multi-keyword matching (combines responses for composite queries)
  - Budget-aware responses (detects ₹ amounts in user input)
  - Broader keyword aliases
"""

import random
import re

# ─────────────────────────────────────────
# Response Templates
# ─────────────────────────────────────────

RESPONSES = {
    "beach": [
        "🏖️ For a beach trip, try **Goa**, **Andaman Islands**, or **Kovalam**! Best visited in Winter.",
        "🌊 Beach lover? **Varkala**, **Tarkarli**, and **Puducherry** offer stunning coastal vibes!",
        "🐚 Don't miss **Gokarna** for a laid-back backpacker beach vibe, or **Vizag** for the Araku Valley combo!",
    ],
    "adventure": [
        "🏔️ Craving adventure? **Leh Ladakh**, **Spiti Valley**, or **Kasol** are perfect for thrill-seekers!",
        "⛺ **Rishikesh** has white-water rafting and trekking — a must for adventurers!",
        "🧗 **Hampta Pass**, **Sandakphu**, or **Nubra Valley** offer world-class trekking experiences!",
    ],
    "cultural": [
        "🏛️ For rich culture, explore **Rajasthan**, **Varanasi**, or **Hampi** — India's heritage gems!",
        "🎭 **Udaipur** and **Jaipur** offer royal palaces and vibrant festivals all winter!",
        "🕌 **Khajuraho**, **Ellora**, and **Tawang** are UNESCO and cultural treasures worth exploring!",
    ],
    "nature": [
        "🌿 Nature lover? **Kerala**, **Coorg**, or **Valley of Flowers** will blow your mind!",
        "🌸 **Munnar** tea gardens and **Sikkim** mountain views are simply breathtaking!",
        "🏔️ **Pangong Lake**, **Lachen**, or **Mawlynnong** are nature's finest masterpieces in India!",
    ],
    "wildlife": [
        "🐯 For wildlife, **Ranthambore** and **Jim Corbett** are India's best tiger reserves!",
        "🦏 **Kaziranga** in Assam is famous for one-horned rhinos — a UNESCO treasure!",
        "🐘 **Kabini**, **Wayanad**, and **Bandipur** offer fantastic elephant and leopard sightings!",
    ],
    "city": [
        "🏙️ **Bangalore** offers a vibrant nightlife, **Hyderabad** has amazing biryani and Charminar!",
        "🌆 **Mumbai** for Bollywood vibes, **Delhi** for history, **Pune** for a relaxed city break!",
        "🍜 **Chennai** and **Ahmedabad** are underrated city destinations with great food and culture!",
    ],
    "budget": [
        "💰 On a tight budget? **Rishikesh** (~₹7,000), **Ooty** (~₹8,000), or **Haridwar** (~₹7,000) are affordable!",
        "🎒 Budget travel tip: Visit during off-season for cheaper hotels and fewer crowds!",
        "💸 **Yelagiri** (₹6,000) and **Tamhini Ghat** (₹6,000) are India's most affordable getaways!",
    ],
    "winter": [
        "❄️ Winter (Nov–Feb) is perfect for **Goa**, **Rajasthan**, **Kerala**, and **Andaman**!",
        "🌅 **Varanasi Ganga Aarti** in winter is a spiritual experience like no other!",
        "🏰 **Udaipur** and **Jaipur** are most magical in winter — cool weather, vibrant festivals!",
    ],
    "summer": [
        "☀️ Escape the heat at **Manali**, **Shimla**, **Leh Ladakh**, or **Darjeeling** in summer!",
        "🏔️ Hill stations are best in summer — cool breeze and lush greenery await!",
        "🌄 **Kasol**, **McLeod Ganj**, and **Spiti Valley** are perfect summer trekking destinations!",
    ],
    "monsoon": [
        "🌧️ Monsoon magic: **Kerala**, **Coorg**, **Munnar**, and **Valley of Flowers** are stunning!",
        "🌈 Waterfalls are at their peak in monsoon — **Athirappilly**, **Malshej Ghat** are must-visits!",
        "☔ **Cherrapunji**, **Agumbe**, and **Bhandardara** transform into paradise during the rains!",
    ],
    "hello": [
        "👋 Hello! I'm BrahamAI, your smart travel companion. Where do you want to explore?",
        "🌍 Hi there! Ask me about beach, adventure, cultural, or nature destinations!",
        "✈️ Hey! Tell me your travel style, season, and budget — I'll find the perfect destination!",
    ],
    "help": [
        "💡 You can ask me:\n- 'Suggest a beach trip'\n- 'Best places in winter'\n- 'Adventure on a budget'\n- 'Wildlife destinations'\n- 'Monsoon travel ideas'",
        "🗺️ I can help with: Beach · Adventure · Cultural · Nature · Wildlife · City trips. Just ask!",
    ],
    "recommend": [
        "🤖 For personalised picks, use the **Get Recommendations** tab or the search form on the home page!",
        "✨ Head to the main page and set your budget, travel style, and season — I'll rank destinations by AI match!",
    ],
    "default": [
        "🤔 Hmm, I didn't catch that. Try asking about a travel type like 'beach', 'adventure', or 'cultural'!",
        "💬 I'm still learning! Try: 'suggest adventure trips' or 'best winter destinations'.",
        "🌐 Ask me about: beach, adventure, nature, cultural, wildlife, city, budget tips, or seasons!",
    ],
}

# Keyword aliases for fuzzy matching
ALIASES = {
    "beach":     ["beach", "sea", "coast", "ocean", "shore", "surf", "snorkel", "dive"],
    "adventure": ["adventure", "trek", "trekking", "hike", "hiking", "rafting", "ski", "climb", "backpack"],
    "cultural":  ["cultural", "culture", "heritage", "history", "temple", "fort", "monument", "unesco"],
    "nature":    ["nature", "forest", "waterfall", "hills", "lake", "valley", "garden", "tea"],
    "wildlife":  ["wildlife", "tiger", "safari", "elephant", "wildlife", "national park", "rhino", "leopard"],
    "city":      ["city", "urban", "metro", "nightlife", "food tour", "street food"],
    "budget":    ["budget", "cheap", "affordable", "low cost", "inexpensive", "economy"],
    "winter":    ["winter", "cold", "december", "january", "february", "nov", "december"],
    "summer":    ["summer", "hot", "may", "june", "july", "april", "hill station"],
    "monsoon":   ["monsoon", "rain", "rainy", "august", "waterfall", "green"],
    "recommend": ["recommend", "suggest", "find", "show me", "give me", "what should"],
    "city":      ["city", "bangalore", "hyderabad", "mumbai", "delhi", "pune", "chennai", "ahmedabad"],
}


def _detect_budget(text: str):
    """Extract a numeric budget value from the input string."""
    # Match patterns like ₹15000, 15000, 15,000, 15k, 15K
    match = re.search(r'₹?\s?(\d[\d,]*)\s*[kK]?', text)
    if match:
        raw = match.group(1).replace(",", "")
        value = int(raw)
        # Handle "15k" → 15000
        if re.search(r'\d+\s*[kK]', text):
            value *= 1000
        return value if value >= 1000 else None
    return None


def _budget_reply(budget: int) -> str:
    """Generate a budget-specific response."""
    if budget <= 8000:
        picks = "Rishikesh, Ooty, Haridwar, Yelagiri, or Tamhini Ghat"
        tier = "budget-friendly"
    elif budget <= 15000:
        picks = "Goa, Jaipur, Munnar, Darjeeling, or Varkala"
        tier = "mid-range"
    elif budget <= 25000:
        picks = "Leh Ladakh, Spiti Valley, Sikkim, or Andaman Islands"
        tier = "premium"
    else:
        picks = "Andaman Islands, Zanskar, Nubra Valley, or Tawang"
        tier = "luxury"
    return f"💰 With ₹{budget:,} you can enjoy {tier} experiences — try {picks}. Use the home search to get AI-ranked results!"


# ─────────────────────────────────────────
# Chatbot Response Function
# ─────────────────────────────────────────

def chatbot_response(user_input: str) -> str:
    """
    Match user input keywords to response templates.
    Handles multi-keyword queries and budget detection.
    Returns a helpful travel suggestion string.
    """
    text = user_input.lower().strip()

    # ── Greeting ──
    if any(k in text for k in ["hello", "hi", "hey", "greet"]):
        return random.choice(RESPONSES["hello"])

    # ── Help ──
    if any(k in text for k in ["help", "what can", "how do", "guide"]):
        return random.choice(RESPONSES["help"])

    # ── Budget detection ──
    detected_budget = _detect_budget(text)

    # ── Multi-keyword matching ──
    matched_keys = []
    for key, aliases in ALIASES.items():
        if any(alias in text for alias in aliases):
            if key not in matched_keys:
                matched_keys.append(key)

    if matched_keys:
        responses = []
        for key in matched_keys[:2]:   # cap at 2 combined topics
            if key in RESPONSES:
                responses.append(random.choice(RESPONSES[key]))

        # Append budget info if a number was also detected
        if detected_budget:
            responses.append(_budget_reply(detected_budget))

        return "\n\n".join(responses)

    # ── Budget-only query ──
    if detected_budget:
        return _budget_reply(detected_budget)

    # ── Default fallback ──
    return random.choice(RESPONSES["default"])
