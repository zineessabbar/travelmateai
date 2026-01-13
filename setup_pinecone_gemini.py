"""
TravelMate AI - Pinecone Setup for Gemini (768 dimensions)
Full migration from OpenAI to Gemini with security and error handling
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Configuration Constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "travelmate-gemini")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Gemini Configuration
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_DIMENSION = 768

# Rate Limiting 
RATE_LIMIT_DELAY = 1.0  
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Initialize APIs
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)

def create_gemini_embedding(text: str, task_type: str = "retrieval_document") -> Optional[List[float]]:
    """
    Create embedding using Gemini with retry logic

    Args:
        text: Text to embed
        task_type: "retrieval_document" for storing, "retrieval_query" for searching

    Returns:
        List of floats (768 dimensions) or None if failed
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type=task_type
            )
            embedding = result['embedding']

            # Validate dimension
            if len(embedding) != EMBEDDING_DIMENSION:
                print(f"⚠️  Warning: Expected {EMBEDDING_DIMENSION} dims, got {len(embedding)}")

            return embedding

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  ⚠️  Retry {attempt + 1}/{MAX_RETRIES} after error: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"  ❌ Failed after {MAX_RETRIES} attempts: {e}")
                return None

    return None


def serialize_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert metadata to Pinecone-compatible format
    Pinecone supports: strings, numbers, booleans, lists of strings

    Args:
        data: Raw metadata dictionary

    Returns:
        Cleaned metadata dictionary
    """
    cleaned = {}

    for key, value in data.items():
        if value is None:
            continue

        # Convert lists to JSON strings (except list of strings)
        if isinstance(value, list):
            if all(isinstance(x, str) for x in value):
                cleaned[key] = value  # List of strings is OK
            elif all(isinstance(x, (int, float)) for x in value):
                cleaned[key] = [str(x) for x in value]  # Convert numbers to strings
            else:
                cleaned[key] = json.dumps(value)  # Complex lists to JSON

        # Keep primitives
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value

        # Convert dicts to JSON
        elif isinstance(value, dict):
            cleaned[key] = json.dumps(value)

        # Convert other types to string
        else:
            cleaned[key] = str(value)

    return cleaned

def create_index() -> Any:
    """Create new Pinecone index for Gemini with proper error handling"""
    existing_indexes = [index.name for index in pc.list_indexes()]

    if INDEX_NAME in existing_indexes:
        # Show current stats
        temp_index = pc.Index(INDEX_NAME)
        stats = temp_index.describe_index_stats()
        print(f"\n Existing index '{INDEX_NAME}' found:")
        print(f"  • Vectors: {stats.total_vector_count}")
        print(f"  • Dimension: {stats.dimension}")

        print(f"\n  Delete and recreate index '{INDEX_NAME}'? (y/n)")
        if input().lower() == 'y':
            pc.delete_index(INDEX_NAME)
            print(f"  Deleted old index")
            time.sleep(5)
        else:
            print(" Using existing index")
            return temp_index

    print(f"\n Creating new index: {INDEX_NAME}")
    print(f"  • Dimensions: {EMBEDDING_DIMENSION}")
    print(f"  • Metric: cosine")
    print(f"  • Cloud: {PINECONE_CLOUD}")
    print(f"  • Region: {PINECONE_REGION}")

    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        )
    )

    print(f" Created index: {INDEX_NAME}")
    print(" Waiting for index to initialize...")
    time.sleep(10)

    return pc.Index(INDEX_NAME)

def prepare_destinations_data() -> List[Dict[str, Any]]:
    """Prepare comprehensive destination data"""
    destinations = [
        {
            "id": "dest_paris_001",
            "name": "Paris",
            "country": "France",
            "continent": "Europe",
            "description": "The City of Light, known for its art, fashion, gastronomy, and culture. Home to iconic landmarks like the Eiffel Tower and Louvre Museum.",
            "vibes": ["romantic", "cultural", "artistic", "historic", "cosmopolitan"],
            "best_for": ["couples", "art_lovers", "food_enthusiasts", "photographers"],
            "avg_cost_per_day_usd": 150,
            "best_months": [4, 5, 6, 9, 10],
            "must_see": ["Eiffel Tower", "Louvre", "Notre-Dame", "Sacré-Cœur"],
            "language": "French",
            "english_friendly": 3.5
        },
        {
            "id": "dest_bali_001",
            "name": "Bali",
            "country": "Indonesia",
            "continent": "Asia",
            "description": "Tropical paradise with stunning beaches, ancient temples, lush rice terraces, and vibrant culture. Perfect for relaxation and spiritual experiences.",
            "vibes": ["tropical", "spiritual", "relaxing", "adventure", "beach"],
            "best_for": ["beach_lovers", "surfers", "yogis", "couples", "backpackers"],
            "avg_cost_per_day_usd": 60,
            "best_months": [4, 5, 6, 7, 8, 9],
            "must_see": ["Tanah Lot", "Ubud Rice Terraces", "Uluwatu Temple", "Seminyak Beach"],
            "language": "Indonesian",
            "english_friendly": 4.2
        },
        {
            "id": "dest_tokyo_001",
            "name": "Tokyo",
            "country": "Japan",
            "continent": "Asia",
            "description": "Ultra-modern metropolis blending cutting-edge technology with traditional culture. From neon-lit skyscrapers to peaceful temples and gardens.",
            "vibes": ["modern", "tech", "cultural", "foodie", "efficient"],
            "best_for": ["tech_enthusiasts", "foodies", "culture_seekers", "photographers"],
            "avg_cost_per_day_usd": 120,
            "best_months": [3, 4, 5, 10, 11],
            "must_see": ["Senso-ji Temple", "Shibuya Crossing", "Tsukiji Market", "Mount Fuji"],
            "language": "Japanese",
            "english_friendly": 3.2
        },
        {
            "id": "dest_marrakech_001",
            "name": "Marrakech",
            "country": "Morocco",
            "continent": "Africa",
            "description": "Exotic desert city with bustling souks, stunning riads, vibrant colors, and rich Berber culture. A sensory adventure in North Africa.",
            "vibes": ["exotic", "cultural", "colorful", "historic", "adventure"],
            "best_for": ["culture_seekers", "photographers", "shoppers", "food_lovers"],
            "avg_cost_per_day_usd": 70,
            "best_months": [3, 4, 5, 10, 11],
            "must_see": ["Jemaa el-Fnaa", "Majorelle Garden", "Bahia Palace", "Medina"],
            "language": "Arabic, French",
            "english_friendly": 3.0
        },
        {
            "id": "dest_santorini_001",
            "name": "Santorini",
            "country": "Greece",
            "continent": "Europe",
            "description": "Stunning Greek island famous for white-washed buildings, blue-domed churches, dramatic cliffs, and breathtaking sunsets over the caldera.",
            "vibes": ["romantic", "scenic", "relaxing", "luxurious", "photogenic"],
            "best_for": ["honeymooners", "couples", "photographers", "luxury_travelers"],
            "avg_cost_per_day_usd": 200,
            "best_months": [5, 6, 9, 10],
            "must_see": ["Oia Sunset", "Red Beach", "Akrotiri", "Wine Tours"],
            "language": "Greek",
            "english_friendly": 4.3
        },
        {
            "id": "dest_dubai_001",
            "name": "Dubai",
            "country": "UAE",
            "continent": "Asia",
            "description": "Futuristic desert metropolis with record-breaking skyscrapers, luxury shopping, pristine beaches, and extravagant experiences.",
            "vibes": ["luxurious", "modern", "shopping", "beach", "futuristic"],
            "best_for": ["luxury_travelers", "shoppers", "families", "beach_lovers"],
            "avg_cost_per_day_usd": 180,
            "best_months": [11, 12, 1, 2, 3],
            "must_see": ["Burj Khalifa", "Dubai Mall", "Palm Jumeirah", "Desert Safari"],
            "language": "Arabic, English",
            "english_friendly": 4.8
        },
        {
            "id": "dest_istanbul_001",
            "name": "Istanbul",
            "country": "Turkey",
            "continent": "Europe/Asia",
            "description": "Historic city bridging Europe and Asia, with Byzantine and Ottoman treasures, bustling bazaars, and incredible cuisine.",
            "vibes": ["historic", "cultural", "vibrant", "exotic", "foodie"],
            "best_for": ["history_buffs", "culture_seekers", "food_lovers", "photographers"],
            "avg_cost_per_day_usd": 60,
            "best_months": [4, 5, 6, 9, 10],
            "must_see": ["Hagia Sophia", "Blue Mosque", "Grand Bazaar", "Bosphorus"],
            "language": "Turkish",
            "english_friendly": 3.5
        },
        {
            "id": "dest_lisbon_001",
            "name": "Lisbon",
            "country": "Portugal",
            "continent": "Europe",
            "description": "Charming coastal capital with colorful tiles, historic trams, delicious pastéis de nata, and stunning viewpoints over the Tagus River.",
            "vibes": ["coastal", "cultural", "historic", "relaxed", "colorful"],
            "best_for": ["culture_seekers", "foodies", "photographers", "budget_travelers"],
            "avg_cost_per_day_usd": 80,
            "best_months": [4, 5, 6, 9, 10],
            "must_see": ["Belém Tower", "Alfama", "Tram 28", "Jerónimos Monastery"],
            "language": "Portuguese",
            "english_friendly": 4.0
        },
        {
            "id": "dest_casablanca_001",
            "name": "Casablanca",
            "country": "Morocco",
            "continent": "Africa",
            "description": "Morocco's economic capital, blending French colonial heritage with modern Moroccan culture. Home to the stunning Hassan II Mosque.",
            "vibes": ["modern", "cultural", "business", "coastal", "cosmopolitan"],
            "best_for": ["business_travelers", "culture_seekers", "architecture_lovers"],
            "avg_cost_per_day_usd": 80,
            "best_months": [4, 5, 6, 9, 10, 11],
            "must_see": ["Hassan II Mosque", "Corniche", "Old Medina", "Morocco Mall"],
            "language": "Arabic, French",
            "english_friendly": 3.2
        },
        {
            "id": "dest_zanzibar_001",
            "name": "Zanzibar",
            "country": "Tanzania",
            "continent": "Africa",
            "description": "Spice island paradise with pristine white beaches, turquoise waters, Stone Town's history, and unique Swahili culture.",
            "vibes": ["beach", "tropical", "cultural", "relaxing", "exotic"],
            "best_for": ["beach_lovers", "honeymooners", "divers", "culture_seekers"],
            "avg_cost_per_day_usd": 100,
            "best_months": [6, 7, 8, 9, 10],
            "must_see": ["Stone Town", "Nungwi Beach", "Spice Tour", "Prison Island"],
            "language": "Swahili, English",
            "english_friendly": 4.0
        }
    ]
    
    return destinations

def prepare_attractions_data() -> List[Dict[str, Any]]:
    """Prepare attractions data"""
    attractions = [
        # Paris
        {
            "id": "attr_louvre_001",
            "name": "Musée du Louvre",
            "destination_id": "dest_paris_001",
            "city": "Paris",
            "category": ["museum", "art", "history"],
            "description": "World's largest art museum housing the Mona Lisa and thousands of masterpieces.",
            "opening_hours": "9h-18h, closed Tuesday",
            "price_usd": 20,
            "duration_hours": 3,
            "tips": "Book online to skip queues. Wednesday evenings less crowded."
        },
        {
            "id": "attr_eiffel_001",
            "name": "Eiffel Tower",
            "destination_id": "dest_paris_001",
            "city": "Paris",
            "category": ["landmark", "viewpoint"],
            "description": "Iconic 324m iron tower offering panoramic views of Paris.",
            "opening_hours": "9:30-23:45 daily",
            "price_usd": 28,
            "duration_hours": 2,
            "tips": "Book skip-the-line tickets. Visit at sunset for best views."
        },
        # Bali
        {
            "id": "attr_tanah_lot_001",
            "name": "Tanah Lot Temple",
            "destination_id": "dest_bali_001",
            "city": "Bali",
            "category": ["temple", "scenic"],
            "description": "Ancient Hindu temple perched on rock formation surrounded by ocean.",
            "opening_hours": "7:00-19:00 daily",
            "price_usd": 4,
            "duration_hours": 1.5,
            "tips": "Best visited at sunset. Wear sarong (provided at entrance)."
        },
        {
            "id": "attr_ubud_rice_001",
            "name": "Tegallalang Rice Terraces",
            "destination_id": "dest_bali_001",
            "city": "Ubud",
            "category": ["nature", "scenic"],
            "description": "Stunning emerald rice paddies carved into hillsides.",
            "opening_hours": "8:00-18:00",
            "price_usd": 3,
            "duration_hours": 2,
            "tips": "Early morning for best light. Try the jungle swings."
        },
        # Marrakech
        {
            "id": "attr_jemaa_001",
            "name": "Jemaa el-Fnaa",
            "destination_id": "dest_marrakech_001",
            "city": "Marrakech",
            "category": ["square", "cultural"],
            "description": "UNESCO square with snake charmers, food stalls, and performers.",
            "opening_hours": "24/7",
            "price_usd": 0,
            "duration_hours": 2,
            "tips": "Visit at sunset. Negotiate prices before eating."
        },
        {
            "id": "attr_majorelle_001",
            "name": "Majorelle Garden",
            "destination_id": "dest_marrakech_001",
            "city": "Marrakech",
            "category": ["garden", "art"],
            "description": "Stunning cobalt blue garden created by Jacques Majorelle, restored by YSL.",
            "opening_hours": "8:00-18:00",
            "price_usd": 10,
            "duration_hours": 1.5,
            "tips": "Visit early to avoid crowds. Combined ticket with YSL museum."
        },
        
    ]
    
    return attractions

def prepare_transport_data() -> List[Dict[str, Any]]:
    """Prepare transport information"""
    transport_info = [
        {
            "id": "transport_paris_001",
            "city": "Paris",
            "destination_id": "dest_paris_001",
            "metro_info": "14 lines. Single ticket €2.15, day pass €13.95",
            "taxi_info": "CDG airport €50-70. Uber available.",
            "tips": "Buy carnet (10 tickets) for savings. Citymapper app essential."
        },
        {
            "id": "transport_bali_001",
            "city": "Bali",
            "destination_id": "dest_bali_001",
            "taxi_info": "Use Grab/Gojek apps. Airport to Seminyak $10-15",
            "scooter_info": "$5/day rental, needs international license",
            "tips": "Grab safest option. Blue Bird taxis reliable."
        },
        {
            "id": "transport_marrakech_001",
            "city": "Marrakech",
            "destination_id": "dest_marrakech_001",
            "taxi_info": "Petit taxis (beige) in city. ~100 DH to airport",
            "bus_info": "ALSA buses modern. Line 19 to airport.",
            "tips": "Always insist on meter. Uber/Careem available."
        },
        {
            "id": "transport_casablanca_001",
            "city": "Casablanca",
            "destination_id": "dest_casablanca_001",
            "taxi_info": "Red petit taxis for city. Grand taxis for longer trips.",
            "tram_info": "Modern tram network, 8 DH per ticket",
            "tips": "Casa Tramway app for routes. Train to Mohammed V airport."
        }
    ]
    
    return transport_info

def create_sample_profiles() -> List[Dict[str, Any]]:
    """Create sample user profiles"""
    profiles = [
        {
            "id": "user_budget_001",
            "user_id": "budget_traveler",
            "travel_style": ["budget", "backpacking", "adventure"],
            "budget_range_min": 30,
            "budget_range_max": 60,
            "interests": ["hostels", "street_food", "local_culture"],
            "visited": []
        },
        {
            "id": "user_luxury_001",
            "user_id": "luxury_traveler",
            "travel_style": ["luxury", "romantic", "relaxing"],
            "budget_range_min": 200,
            "budget_range_max": 500,
            "interests": ["fine_dining", "spa", "boutique_hotels"],
            "visited": ["Paris", "Dubai"]
        },
        {
            "id": "user_family_001",
            "user_id": "family_traveler",
            "travel_style": ["family", "educational", "safe"],
            "budget_range_min": 100,
            "budget_range_max": 200,
            "interests": ["kid_activities", "museums", "beaches"],
            "visited": []
        }
    ]

    return profiles


def save_checkpoint(failed_vectors: List[Dict], checkpoint_file: str = "checkpoint_failed.json"):
    """Save failed vectors to checkpoint file"""
    with open(checkpoint_file, 'w') as f:
        json.dump(failed_vectors, f, indent=2)
    print(f" Saved {len(failed_vectors)} failed vectors to {checkpoint_file}")

def populate_pinecone(index: Any) -> Dict[str, int]:
    """
    Populate Pinecone with Gemini embeddings

    Returns:
        Statistics dictionary
    """
    print("\n Starting Pinecone population with Gemini embeddings...")

    destinations = prepare_destinations_data()
    attractions = prepare_attractions_data()
    transport_info = prepare_transport_data()
    profiles = create_sample_profiles()

    all_vectors = []
    failed_items = []
    stats = {"total": 0, "success": 0, "failed": 0}

    # Process destinations
    print("\n Processing destinations...")
    for dest in destinations:
        stats["total"] += 1
        text = f"{dest['name']} {dest['country']} {dest['description']} {' '.join(dest['vibes'])}"

        embedding = create_gemini_embedding(text)
        if embedding:
            # Serialize metadata properly
            metadata = serialize_metadata({
                **dest,
                "type": "destination",
                "created_at": datetime.now().isoformat()
            })

            vector = {
                "id": dest["id"],
                "values": embedding,
                "metadata": metadata
            }
            all_vectors.append(vector)
            stats["success"] += 1
            print(f"   {dest['name']}, {dest['country']}")
        else:
            stats["failed"] += 1
            failed_items.append({"type": "destination", "data": dest})
            print(f"   Failed: {dest['name']}")
            continue

        time.sleep(RATE_LIMIT_DELAY)

    # Process attractions
    print("\n Processing attractions...")
    for attr in attractions:
        stats["total"] += 1
        text = f"{attr['name']} {attr['description']} {' '.join(attr['category'])}"

        embedding = create_gemini_embedding(text)
        if embedding:
            metadata = serialize_metadata({
                **attr,
                "type": "attraction",
                "created_at": datetime.now().isoformat()
            })

            vector = {
                "id": attr["id"],
                "values": embedding,
                "metadata": metadata
            }
            all_vectors.append(vector)
            stats["success"] += 1
            print(f"   {attr['name']} ({attr['city']})")
        else:
            stats["failed"] += 1
            failed_items.append({"type": "attraction", "data": attr})
            print(f"   Failed: {attr['name']}")
            continue

        time.sleep(RATE_LIMIT_DELAY)

    # Process transport
    print("\n Processing transport info...")
    for transport in transport_info:
        stats["total"] += 1
        text = f"{transport['city']} transport {transport.get('tips', '')}"

        embedding = create_gemini_embedding(text)
        if embedding:
            metadata = serialize_metadata({
                **transport,
                "type": "transport_info",
                "created_at": datetime.now().isoformat()
            })

            vector = {
                "id": transport["id"],
                "values": embedding,
                "metadata": metadata
            }
            all_vectors.append(vector)
            stats["success"] += 1
            print(f"   {transport['city']} transport")
        else:
            stats["failed"] += 1
            failed_items.append({"type": "transport", "data": transport})
            print(f"   Failed: {transport['city']}")
            continue

        time.sleep(RATE_LIMIT_DELAY)

    # Process user profiles
    print("\n👤 Processing user profiles...")
    for profile in profiles:
        stats["total"] += 1
        text = f"User profile: {' '.join(profile['travel_style'])} {' '.join(profile['interests'])}"

        embedding = create_gemini_embedding(text)
        if embedding:
            metadata = serialize_metadata({
                **profile,
                "type": "user_profile",
                "created_at": datetime.now().isoformat()
            })

            vector = {
                "id": profile["id"],
                "values": embedding,
                "metadata": metadata
            }
            all_vectors.append(vector)
            stats["success"] += 1
            print(f"   {profile['user_id']}")
        else:
            stats["failed"] += 1
            failed_items.append({"type": "profile", "data": profile})
            print(f"   Failed: {profile['user_id']}")
            continue

        time.sleep(RATE_LIMIT_DELAY)

    # Upsert to Pinecone with retry logic
    print(f"\n📤 Uploading {len(all_vectors)} vectors to Pinecone...")
    upload_failed = []

    for i in range(0, len(all_vectors), BATCH_SIZE):
        batch = all_vectors[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(all_vectors) - 1) // BATCH_SIZE + 1

        success = False
        for attempt in range(MAX_RETRIES):
            try:
                index.upsert(vectors=batch)
                print(f"  ✅ Batch {batch_num}/{total_batches}")
                success = True
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"  ⚠️  Batch {batch_num} retry {attempt + 1}/{MAX_RETRIES}: {e}")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    print(f"  ❌ Batch {batch_num} failed after {MAX_RETRIES} attempts")
                    upload_failed.extend(batch)

        if success:
            time.sleep(0.5)  # Brief pause between batches

    # Save failed items if any
    if failed_items:
        save_checkpoint(failed_items, "checkpoint_failed_embeddings.json")

    if upload_failed:
        save_checkpoint(upload_failed, "checkpoint_failed_uploads.json")

    print(f"\n Upload complete!")
    print(f"  • Successful vectors: {len(all_vectors) - len(upload_failed)}")
    if upload_failed:
        print(f"  • Failed uploads: {len(upload_failed)} (saved to checkpoint)")

    return stats

def test_search(index: Any):
    """Test semantic search with Gemini embeddings"""
    print("\n🔍 Testing semantic search...")

    test_queries = [
        "romantic beach destination for honeymoon",
        "cheap backpacking adventure in Asia",
        "cultural city in Morocco with markets",
        "luxury desert experience"
    ]

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")

        try:
            # Create query embedding with Gemini
            query_embedding = create_gemini_embedding(query, task_type="retrieval_query")

            if not query_embedding:
                print("  ❌ Failed to create query embedding")
                continue

            # Search
            results = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True,
                filter={"type": "destination"}
            )

            if results.matches:
                print("  Results:")
                for i, match in enumerate(results.matches, 1):
                    name = match.metadata.get('name', 'Unknown')
                    country = match.metadata.get('country', '')
                    score = match.score
                    print(f"    {i}. {name}, {country} (score: {score:.3f})")
            else:
                print("  No results found")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        time.sleep(RATE_LIMIT_DELAY)

def verify_gemini_setup() -> bool:
    """Verify Gemini is working correctly"""
    print("\n Verifying Gemini setup...")

    try:
        # Test embedding
        test_embedding = create_gemini_embedding("test")
        if not test_embedding:
            raise Exception("Failed to create test embedding")

        print(f" Embedding working: {len(test_embedding)} dimensions")

        # Test generation
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content("Say 'Gemini is ready' in 3 words")
        print(f" Generation working: {response.text.strip()}")

        return True

    except Exception as e:
        print(f" Gemini setup error: {e}")
        return False


def verify_environment() -> bool:
    """Verify all required environment variables are set"""
    print("\n Checking environment variables...")

    missing = []

    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")

    if missing:
        print("\n Missing required environment variables:")
        for var in missing:
            print(f"  • {var}")
        print("\n Please:")
        print("  1. Copy .env.example to .env")
        print("  2. Fill in your API keys")
        print("  3. Run this script again")
        return False

    print(" All required variables set")
    return True

def main():
    """Main execution"""
    print("=" * 70)
    print("🌍 TravelMate AI - Full Gemini Setup")
    print("=" * 70)

    # Check environment
    if not verify_environment():
        return

    # Verify Gemini
    if not verify_gemini_setup():
        return

    print("\n" + "=" * 70)
    print("Ready to create Pinecone index with Gemini embeddings")
    print(f"Index: {INDEX_NAME}")
    print(f"Dimensions: {EMBEDDING_DIMENSION}")
    print(f"Cloud: {PINECONE_CLOUD} ({PINECONE_REGION})")
    print("=" * 70)

    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    try:
        # Create index
        index = create_index()

        # Populate with data
        stats = populate_pinecone(index)

        # Test search
        test_search(index)

        # Final stats
        index_stats = index.describe_index_stats()
        print(f"\n Final Statistics:")
        print(f"  Total items processed: {stats['total']}")
        print(f"  Successful embeddings: {stats['success']}")
        print(f"  Failed embeddings: {stats['failed']}")
        print(f"  Vectors in index: {index_stats.total_vector_count}")
        print(f"  Dimension: {index_stats.dimension}")

        print("\n" + "=" * 70)
        print(" Full Gemini setup complete!")
        print("=" * 70)
        print("\n What's been created:")
        print(f"  • Pinecone index '{INDEX_NAME}' with {EMBEDDING_DIMENSION} dimensions")
        print(f"  • {stats['success']} items with Gemini embeddings")
        print("  • Destinations, attractions, transport, and user profiles")
        print("  • Everything ready for your TravelMate AI app")

        if stats['failed'] > 0:
            print(f"\n  Note: {stats['failed']} items failed (check checkpoint files)")

    except Exception as e:
        print(f"\n Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
