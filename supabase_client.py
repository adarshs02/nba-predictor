import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Supabase credentials from environment variables
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

# Initialize and export the Supabase client
if not supabase_url or not supabase_key:
    raise ValueError("Supabase URL and Key must be set in the environment variables.")

supabase: Client = create_client(supabase_url, supabase_key)
