import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import random
import os

def clean_price(price_str):
    # Extract numerical price from string like "Rp 212.000.000" or "-5% Rp 255.000.000"
    if not price_str:
        return None
    price_str = price_str.replace("Rp", "").replace(".", "").replace(",", "").strip()
    match = re.search(r'\d+', price_str)
    if match:
        return int(match.group())
    return None

def clean_mileage(mileage_str):
    # Convert mileage string like "40 - 45K KM" or "8000 KM" to a numerical value (in KM)
    if not mileage_str:
        return None
    mileage_str = mileage_str.upper().replace("KM", "").strip()
    # Case 1: Range like "40 - 45K"
    if "-" in mileage_str:
        parts = mileage_str.split("-")
        vals = []
        for p in parts:
            p = p.strip()
            multiplier = 1
            if "K" in p:
                p = p.replace("K", "")
                multiplier = 1000
            try:
                vals.append(float(p) * multiplier)
            except ValueError:
                pass
        if len(vals) == 2:
            return int(sum(vals) / 2) # Return average of range
    # Case 2: Single value like "8000" or "15K"
    multiplier = 1
    if "K" in mileage_str:
        mileage_str = mileage_str.replace("K", "")
        multiplier = 1000
    try:
        return int(float(mileage_str) * multiplier)
    except ValueError:
        return None

def scrape_carmudi(num_pages=40):
    all_cars = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print(f"Starting to scrape {num_pages} pages from Carmudi Indonesia...")
    
    for page in range(1, num_pages + 1):
        url = f"https://www.carmudi.co.id/mobil-dijual/indonesia?page_number={page}&page_size=25"
        print(f"Scraping Page {page}/{num_pages}: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"  [Warning] Failed to fetch page {page}. Status code: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all card/listing items. Carmudi uses articles and divs with classes containing 'listing'
            # Let's target the listing container cards
            cards = soup.find_all(['article', 'div'], class_=re.compile(r'listing|card'))
            page_cars = 0
            
            for card in cards:
                # Find title
                title_elem = card.find('h2')
                if not title_elem:
                    continue
                title = title_elem.text.strip()
                
                # Check if it starts with a year (e.g. 2023 Hyundai Stargazer)
                match_year = re.match(r'^(\d{4})\s+(.+)$', title)
                if not match_year:
                    continue
                
                year = int(match_year.group(1))
                rest_title = match_year.group(2)
                
                # Exclude header titles or irrelevant sections
                if len(title) > 100 or "temukan" in title.lower() or "momen" in title.lower():
                    continue
                
                # Split brand and model
                # Brand is usually the first word of the rest of the title
                words = rest_title.split()
                if not words:
                    continue
                brand = words[0]
                model = " ".join(words[1:]) if len(words) > 1 else "Unknown"
                
                # Find price
                # Look for price classes or tags containing 'Rp'
                price_text = None
                price_elem = card.find(['div', 'span', 'p'], class_=re.compile(r'price'))
                if price_elem:
                    price_text = price_elem.text.strip()
                else:
                    # Search text for 'Rp' inside the card
                    price_tags = card.find_all(text=re.compile(r'Rp\s?\d+'))
                    if price_tags:
                        price_text = price_tags[0].strip()
                        
                price = clean_price(price_text)
                if not price or price < 10000000: # Exclude invalid prices or DP-only
                    continue
                
                # Find specs (transmission, mileage, location)
                # Typically they are in spans or list items
                transmission = "Unknown"
                mileage = None
                location = "Unknown"
                
                # Extract all text items from elements that look like specs or tags
                specs = []
                for spec_elem in card.find_all(['span', 'li', 'div'], class_=re.compile(r'spec|attr|meta|info|item|parameter|badge|location|transmission|mileage')):
                    txt = spec_elem.text.strip()
                    if txt and len(txt) < 30:
                        specs.append(txt)
                
                # Add default classes scan
                for text_span in card.find_all(['span', 'li']):
                    txt = text_span.text.strip()
                    if txt and len(txt) < 30:
                        specs.append(txt)
                
                # Clean and deduplicate specs list
                specs = list(set(specs))
                
                # Parse specs
                for spec in specs:
                    spec_lower = spec.lower()
                    if "automatic" in spec_lower or "cvt" in spec_lower:
                        transmission = "Automatic"
                    elif "manual" in spec_lower:
                        transmission = "Manual"
                    elif "km" in spec_lower:
                        mileage = clean_mileage(spec)
                    elif any(loc in spec_lower for loc in ["jakarta", "banten", "jawa barat", "jawa tengah", "jawa timur", "yogyakarta", "bali", "sumatera", "riau", "kalimantan", "sulawesi", "tangerang", "bekasi", "depok", "bogor", "bandung", "surabaya", "semarang", "medan"]):
                        location = spec.strip()
                
                # If transmission is still unknown, check title
                if transmission == "Unknown":
                    if "a/t" in title.lower() or "at" in title.lower() or "automatic" in title.lower() or "cvt" in title.lower():
                        transmission = "Automatic"
                    elif "m/t" in title.lower() or "mt" in title.lower() or "manual" in title.lower():
                        transmission = "Manual"
                
                # Standardize location
                if location == "Unknown":
                    # Try to search for location inside the card text
                    for txt_elem in card.find_all(text=True):
                        t_txt = txt_elem.strip()
                        if any(loc in t_txt.lower() for loc in ["jakarta", "banten", "jawa barat", "jawa tengah", "jawa timur", "yogyakarta", "bali", "sumatera", "riau", "kalimantan", "sulawesi"]):
                            location = t_txt
                            break
                            
                # Exclude duplicate cars within this run
                car_id = f"{year}_{brand}_{model}_{price}_{mileage}_{transmission}"
                if any(x['car_id'] == car_id for x in all_cars):
                    continue
                
                # Fuel estimation (helpful for additional features)
                fuel_type = "Bensin"
                model_lower = model.lower()
                brand_lower = brand.lower()
                if "diesel" in model_lower or "2.4" in model_lower or "fortuner" in model_lower or "pajero" in model_lower or "innovas" in model_lower:
                    if any(x in model_lower for x in ["d-4d", "vnt", "trd", "gr", "dakar", "exceed", "glx"]):
                        fuel_type = "Diesel"
                if "ev" in model_lower or "electric" in model_lower or "air ev" in model_lower or "ioniq" in model_lower or "bav" in model_lower:
                    fuel_type = "Elektrik"
                elif "hybrid" in model_lower or "hv" in model_lower or "hev" in model_lower:
                    fuel_type = "Hybrid"
                
                # Append if we have mileage and transmission
                if mileage is not None and transmission != "Unknown":
                    all_cars.append({
                        "car_id": car_id,
                        "brand": brand,
                        "model": model,
                        "year": year,
                        "mileage": mileage,
                        "transmission": transmission,
                        "location": location,
                        "fuel_type": fuel_type,
                        "price": price
                    })
                    page_cars += 1
            
            print(f"  Successfully extracted {page_cars} unique cars from Page {page}.")
            
            # Politeness delay
            time.sleep(random.uniform(1.0, 2.0))
            
        except Exception as e:
            print(f"  [Error] Scraping page {page} failed: {e}")
            time.sleep(3.0)
            
    print(f"Scraping completed. Total cars extracted: {len(all_cars)}")
    
    # Save to CSV
    df = pd.DataFrame(all_cars)
    if not df.empty:
        # Drop the temporary car_id column
        df = df.drop(columns=["car_id"])
        
        # Output directory
        output_dir = "C:\\Users\\USER\\.gemini\\antigravity\\scratch\\proyek_sistem_cerdas"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "mobil_bekas_carmudi.csv")
        
        df.to_csv(output_file, index=False)
        print(f"Dataset successfully saved to: {output_file}")
        print("Data Summary:")
        print(df.info())
        print(df.head())
    else:
        print("[Error] No data collected! Check the scraper HTML selectors.")

if __name__ == "__main__":
    scrape_carmudi(num_pages=40)
