"""
BrahamAI — Simple Rule-Based Chatbot
Provides travel suggestions via keyword matching
"""

import random

# ─────────────────────────────────────────
# Response Templates
# ─────────────────────────────────────────

RESPONSES = {
    "beach": [
        "🏖️ For a beach trip, try **Goa**, **Andaman Islands**, or **Kovalam**! Best visited in Winter.",
        "🌊 Beach lover? **Varkala** and **Puducherry** offer stunning coastal vibes!",
    ],
    "adventure": [
        "🏔️ Craving adventure? **Leh Ladakh**, **Spiti Valley**, or **Kasol** are perfect for thrill-seekers!",
        "⛺ **Rishikesh** has white-water rafting and trekking — a must for adventurers!",
    ],
    "cultural": [
        "🏛️ For rich culture, explore **Rajasthan**, **Varanasi**, or **Hampi** — India's heritage gems!",
        "🎭 **Udaipur** and **Jaipur** offer royal palaces and vibrant festivals all winter!",
    ],
    "nature": [
        "🌿 Nature lover? **Kerala**, **Coorg**, or **Valley of Flowers** will blow your mind!",
        "🌸 **Munnar** tea gardens and **Sikkim** mountain views are simply breathtaking!",
    ],
    "wildlife": [
        "🐯 For wildlife, **Ranthambore** and **Jim Corbett** are India's best tiger reserves!",
        "🦏 **Kaziranga** in Assam is famous for one-horned rhinos — a UNESCO treasure!",
    ],
    "budget": [
        "💰 On a tight budget? **Rishikesh** (~₹7,000), **Ooty** (~₹8,000), or **Haridwar** (~₹7,000) are affordable!",
        "🎒 Budget travel tip: Visit during off-season for cheaper hotels and fewer crowds!",
    ],
    "winter": [
        "❄️ Winter (Nov–Feb) is perfect for **Goa**, **Rajasthan**, **Kerala**, and **Andaman**!",
        "🌅 **Varanasi Ganga Aarti** in winter is a spiritual experience like no other!",
    ],
    "summer": [
        "☀️ Escape the heat at **Manali**, **Shimla**, **Leh Ladakh**, or **Darjeeling** in summer!",
        "🏔️ Hill stations are best in summer — cool breeze and lush greenery await!",
    ],
    "monsoon": [
        "🌧️ Monsoon magic: **Kerala**, **Coorg**, **Munnar**, and **Valley of Flowers** are stunning!",
        "🌈 Waterfalls are at their peak in monsoon — **Dudhsagar**, **Athirappilly** are must-visits!",
    ],
    "hello": [
        "👋 Hello! I'm BrahamAI, your smart travel companion. Where do you want to explore?",
        "🌍 Hi there! Ask me about beach, adventure, cultural, or nature destinations!",
    ],
    "help": [
        "💡 You can ask me:\n- 'Suggest a beach trip'\n- 'Best places in winter'\n- 'Adventure on a budget'\n- 'Wildlife destinations'",
        "🗺️ I can help with: Beach · Adventure · Cultural · Nature · Wildlife trips. Just ask!",
    ],
    "default": [
        "🤔 Hmm, I didn't catch that. Try asking about a travel type like 'beach', 'adventure', or 'cultural'!",
        "💬 I'm still learning! Try: 'suggest adventure trips' or 'best winter destinations'.",
        "🌐 Ask me about: beach, adventure, nature, cultural, wildlife, budget tips, or seasons!",
    ]
}


# ─────────────────────────────────────────
# Chatbot Response Function
# ─────────────────────────────────────────

def chatbot_response(user_input: str) -> str:
    """
    Match user input keywords to response templates.
    Returns a helpful travel suggestion string.
    """
    text = user_input.lower().strip()

    # Priority keyword matching
    for keyword in ["hello", "hi", "hey"]:
        if keyword in text:
            return random.choice(RESPONSES["hello"])

    if "help" in text or "what can" in text:
        return random.choice(RESPONSES["help"])

    for keyword, responses in RESPONSES.items():
        if keyword in text:
            return random.choice(responses)

    return random.choice(RESPONSES["default"])
