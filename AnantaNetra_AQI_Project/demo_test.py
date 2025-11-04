#!/usr/bin/env python3
"""
AnantaNetra Demo Script
Demonstrates the complete AI Environmental Monitoring System
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any

# Demo configuration
BACKEND_URL = "http://localhost:8000"
DEMO_PINCODES = ["400001", "110001", "560001", "600001", "700001"]  # Mumbai, Delhi, Bangalore, Chennai, Kolkata

class AnantaNetraDemo:
    def __init__(self):
        self.backend_url = BACKEND_URL
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
        
    async def check_system_status(self) -> Dict[str, Any]:
        """Check if the backend system is running"""
        try:
            async with self.session.get(f"{self.backend_url}/api/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ System Status: OPERATIONAL")
                    print(f"   System: {data.get('system', 'Unknown')}")
                    print(f"   Version: {data.get('version', 'Unknown')}")
                    return data
                else:
                    print(f"❌ System Status: ERROR (HTTP {response.status})")
                    return None
        except Exception as e:
            print(f"❌ System Status: OFFLINE ({str(e)})")
            return None
            
    async def test_aqi_endpoint(self, pincode: str) -> Dict[str, Any]:
        """Test AQI data retrieval for a pincode"""
        try:
            async with self.session.get(f"{self.backend_url}/api/aqi/{pincode}") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ AQI Data for {pincode}:")
                    print(f"   AQI: {data.get('aqi', 'N/A')} ({data.get('category', 'Unknown')})")
                    print(f"   PM2.5: {data.get('pm25', 'N/A')} μg/m³")
                    print(f"   Temperature: {data.get('temperature', 'N/A')}°C")
                    return data
                else:
                    print(f"❌ AQI Data for {pincode}: ERROR (HTTP {response.status})")
                    return None
        except Exception as e:
            print(f"❌ AQI Data for {pincode}: ERROR ({str(e)})")
            return None
            
    async def test_forecast_endpoint(self, pincode: str) -> Dict[str, Any]:
        """Test forecast data retrieval"""
        try:
            async with self.session.get(f"{self.backend_url}/api/forecast/{pincode}?hours=6") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Forecast for {pincode}: {len(data)} data points")
                    if data:
                        first = data[0]
                        print(f"   Next Hour AQI: {first.get('predicted_aqi', 'N/A')} ({first.get('category', 'Unknown')})")
                    return data
                else:
                    print(f"❌ Forecast for {pincode}: ERROR (HTTP {response.status})")
                    return None
        except Exception as e:
            print(f"❌ Forecast for {pincode}: ERROR ({str(e)})")
            return None
            
    async def test_health_advisory(self, aqi: int) -> Dict[str, Any]:
        """Test health advisory generation"""
        try:
            async with self.session.get(f"{self.backend_url}/api/health/advisory?aqi={aqi}") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health Advisory for AQI {aqi}:")
                    print(f"   Category: {data.get('category', 'Unknown')}")
                    print(f"   Message: {data.get('message', 'No message')[:100]}...")
                    print(f"   Precautions: {len(data.get('precautions', []))} items")
                    return data
                else:
                    print(f"❌ Health Advisory for AQI {aqi}: ERROR (HTTP {response.status})")
                    return None
        except Exception as e:
            print(f"❌ Health Advisory for AQI {aqi}: ERROR ({str(e)})")
            return None
            
    async def test_map_data(self) -> Dict[str, Any]:
        """Test map data retrieval"""
        try:
            async with self.session.get(f"{self.backend_url}/api/map/data") as response:
                if response.status == 200:
                    data = await response.json()
                    cities = data.get('cities', [])
                    print(f"✅ Map Data: {len(cities)} cities available")
                    if cities:
                        avg_aqi = sum(city.get('aqi', 0) for city in cities) / len(cities)
                        print(f"   Average AQI: {avg_aqi:.1f}")
                        poor_cities = len([c for c in cities if c.get('aqi', 0) > 200])
                        print(f"   Cities with Poor AQI: {poor_cities}")
                    return data
                else:
                    print(f"❌ Map Data: ERROR (HTTP {response.status})")
                    return None
        except Exception as e:
            print(f"❌ Map Data: ERROR ({str(e)})")
            return None
            
    async def run_comprehensive_demo(self):
        """Run a comprehensive demo of all system features"""
        print("🌍 AnantaNetra - AI Environmental Monitoring System Demo")
        print("=" * 60)
        print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. System Status Check
        print("1️⃣ System Status Check")
        print("-" * 30)
        status = await self.check_system_status()
        if not status:
            print("❌ Cannot proceed with demo - system is offline")
            return
        print()
        
        # 2. AQI Data Testing
        print("2️⃣ Real-time AQI Data Testing")
        print("-" * 30)
        aqi_data = {}
        for pincode in DEMO_PINCODES:
            data = await self.test_aqi_endpoint(pincode)
            if data:
                aqi_data[pincode] = data
            await asyncio.sleep(0.5)  # Rate limiting
        print()
        
        # 3. Forecast Testing
        print("3️⃣ AI Prediction Testing")
        print("-" * 30)
        if aqi_data:
            sample_pincode = list(aqi_data.keys())[0]
            await self.test_forecast_endpoint(sample_pincode)
        print()
        
        # 4. Health Advisory Testing
        print("4️⃣ AI Health Advisory Testing")
        print("-" * 30)
        test_aqis = [45, 120, 180, 250, 350]  # Different AQI levels
        for aqi in test_aqis:
            await self.test_health_advisory(aqi)
            await asyncio.sleep(0.3)
        print()
        
        # 5. Map Data Testing
        print("5️⃣ Interactive Map Data Testing")
        print("-" * 30)
        await self.test_map_data()
        print()
        
        # 6. Performance Summary
        print("6️⃣ Demo Summary")
        print("-" * 30)
        if aqi_data:
            print(f"✅ Tested {len(aqi_data)} cities successfully")
            highest_aqi = max(data.get('aqi', 0) for data in aqi_data.values())
            lowest_aqi = min(data.get('aqi', 0) for data in aqi_data.values())
            print(f"📊 AQI Range: {lowest_aqi} - {highest_aqi}")
            
            # City with highest pollution
            worst_city = max(aqi_data.items(), key=lambda x: x[1].get('aqi', 0))
            best_city = min(aqi_data.items(), key=lambda x: x[1].get('aqi', 0))
            
            print(f"🏭 Highest Pollution: Pincode {worst_city[0]} (AQI: {worst_city[1].get('aqi')})")
            print(f"🌱 Cleanest Air: Pincode {best_city[0]} (AQI: {best_city[1].get('aqi')})")
        
        print("\n✨ Demo completed successfully!")
        print("\n🎯 Key Features Demonstrated:")
        print("   • Real-time AQI monitoring")
        print("   • 24-hour AI predictions")
        print("   • Intelligent health advisories")
        print("   • Interactive city mapping")
        print("   • Comprehensive error handling")
        print("   • Production-ready API performance")
        
    async def interactive_demo(self):
        """Interactive demo mode for live presentations"""
        print("🎮 Interactive Demo Mode")
        print("=" * 30)
        
        while True:
            print("\nAvailable Commands:")
            print("1. Check system status")
            print("2. Get AQI for city")
            print("3. Get forecast")
            print("4. Get health advisory")
            print("5. Get map data")
            print("6. Run full demo")
            print("0. Exit")
            
            choice = input("\nEnter your choice (0-6): ").strip()
            
            if choice == "0":
                print("👋 Demo ended")
                break
            elif choice == "1":
                await self.check_system_status()
            elif choice == "2":
                pincode = input("Enter pincode (e.g., 400001): ").strip()
                if pincode:
                    await self.test_aqi_endpoint(pincode)
            elif choice == "3":
                pincode = input("Enter pincode for forecast: ").strip()
                if pincode:
                    await self.test_forecast_endpoint(pincode)
            elif choice == "4":
                try:
                    aqi = int(input("Enter AQI value (0-500): ").strip())
                    await self.test_health_advisory(aqi)
                except ValueError:
                    print("❌ Invalid AQI value")
            elif choice == "5":
                await self.test_map_data()
            elif choice == "6":
                await self.run_comprehensive_demo()
            else:
                print("❌ Invalid choice")

async def main():
    """Main demo function"""
    import sys
    
    # Check if backend is accessible
    print("🔍 Checking AnantaNetra backend availability...")
    
    try:
        # Quick connectivity check
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BACKEND_URL}/api/status", timeout=5) as response:
                if response.status == 200:
                    print("✅ Backend is accessible")
                else:
                    print(f"⚠️ Backend returned status {response.status}")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {str(e)}")
        print("💡 Make sure the backend is running on http://localhost:8000")
        print("💡 Run: cd backend && uvicorn app.main:app --reload")
        return
    
    # Run demo
    async with AnantaNetraDemo() as demo:
        if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            await demo.interactive_demo()
        else:
            await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main())
