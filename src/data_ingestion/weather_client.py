"""
data_ingestion/weather_client.py
Fetches real-time + forecast weather from Open-Meteo (free, no API key).
Falls back to synthetic weather when offline or rate-limited.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WeatherReading:
    timestamp: str
    temperature_c: float
    precipitation_mm_h: float
    wind_speed_kmh: float
    condition: str          # "clear" | "light_rain" | "heavy_rain" | "hot"
    is_forecast: bool = False


class WeatherClient:
    """
    Fetches weather data for a city location.
    Primary: Open-Meteo API (free, no key required).
    Fallback: Synthetic weather generator.

    Usage:
        client = WeatherClient(lat=18.52, lon=73.85)  # Pune
        current = client.get_current()
        forecast = client.get_forecast_24h()
    """

    OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, lat: float = 18.52, lon: float = 73.85,
                 cache_ttl_seconds: int = 900):
        self.lat = lat
        self.lon = lon
        self.cache_ttl = cache_ttl_seconds
        self._cache: Optional[Dict] = None
        self._cache_time: float = 0.0
        self._rng = np.random.default_rng(42)

    # ── Public API ────────────────────────────────────────────────────────────
    def get_current(self) -> WeatherReading:
        """Returns current weather reading."""
        data = self._fetch_or_fallback()
        return self._parse_current(data)

    def get_forecast_24h(self) -> List[WeatherReading]:
        """Returns list of 24 hourly forecast readings."""
        data = self._fetch_or_fallback()
        return self._parse_forecast(data)

    def get_condition_string(self) -> str:
        """Returns simple condition string for simulation environment."""
        try:
            reading = self.get_current()
            return reading.condition
        except Exception:
            return "clear"

    # ── Fetch ─────────────────────────────────────────────────────────────────
    def _fetch_or_fallback(self) -> Dict:
        # Return cached data if fresh
        if self._cache and (time.time() - self._cache_time) < self.cache_ttl:
            return self._cache

        try:
            data = self._fetch_open_meteo()
            self._cache = data
            self._cache_time = time.time()
            logger.debug("Weather fetched from Open-Meteo API")
            return data
        except Exception as e:
            logger.warning(f"Weather API unavailable ({e}), using synthetic fallback")
            return self._synthetic_fallback()

    def _fetch_open_meteo(self) -> Dict:
        import urllib.request, json
        params = (
            f"latitude={self.lat}&longitude={self.lon}"
            f"&current=temperature_2m,precipitation,wind_speed_10m"
            f"&hourly=temperature_2m,precipitation,wind_speed_10m"
            f"&forecast_days=1&timezone=auto"
        )
        url = f"{self.OPEN_METEO_URL}?{params}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read())

    def _synthetic_fallback(self) -> Dict:
        """Generate plausible weather data without API access."""
        import datetime
        hour = datetime.datetime.now().hour
        # Diurnal temperature curve
        temp = 22 + 8 * np.sin(np.pi * (hour - 6) / 12)
        precip = float(self._rng.exponential(0.3) if self._rng.random() < 0.15 else 0.0)
        wind = float(self._rng.uniform(5, 25))

        hourly_temps = [22 + 8 * np.sin(np.pi * (h - 6) / 12) + self._rng.normal(0, 1)
                        for h in range(24)]
        hourly_precip = [float(self._rng.exponential(0.3) if self._rng.random() < 0.15 else 0.0)
                         for _ in range(24)]

        return {
            "_synthetic": True,
            "current": {
                "temperature_2m": round(float(temp), 1),
                "precipitation": round(float(precip), 2),
                "wind_speed_10m": round(float(wind), 1),
            },
            "hourly": {
                "time": [f"2024-01-01T{h:02d}:00" for h in range(24)],
                "temperature_2m": [round(t, 1) for t in hourly_temps],
                "precipitation": [round(p, 2) for p in hourly_precip],
                "wind_speed_10m": [round(float(self._rng.uniform(5, 25)), 1)
                                   for _ in range(24)],
            }
        }

    # ── Parsers ───────────────────────────────────────────────────────────────
    def _parse_current(self, data: Dict) -> WeatherReading:
        import datetime
        cur = data.get("current", {})
        temp = cur.get("temperature_2m", 22.0)
        precip = cur.get("precipitation", 0.0)
        wind = cur.get("wind_speed_10m", 10.0)
        condition = self._classify(temp, precip)
        return WeatherReading(
            timestamp=datetime.datetime.now().isoformat(),
            temperature_c=float(temp),
            precipitation_mm_h=float(precip),
            wind_speed_kmh=float(wind),
            condition=condition,
            is_forecast=False,
        )

    def _parse_forecast(self, data: Dict) -> List[WeatherReading]:
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        precips = hourly.get("precipitation", [])
        winds = hourly.get("wind_speed_10m", [])

        readings = []
        for i in range(min(24, len(times))):
            t = temps[i] if i < len(temps) else 22.0
            p = precips[i] if i < len(precips) else 0.0
            w = winds[i] if i < len(winds) else 10.0
            readings.append(WeatherReading(
                timestamp=times[i],
                temperature_c=float(t),
                precipitation_mm_h=float(p),
                wind_speed_kmh=float(w),
                condition=self._classify(t, p),
                is_forecast=True,
            ))
        return readings

    @staticmethod
    def _classify(temp: float, precip: float) -> str:
        if precip > 5.0:
            return "heavy_rain"
        elif precip > 0.5:
            return "light_rain"
        elif temp > 35:
            return "hot"
        return "clear"


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = WeatherClient(lat=18.52, lon=73.85)
    current = client.get_current()
    print(f"Current: {current.temperature_c}°C, {current.condition}, "
          f"precip={current.precipitation_mm_h}mm/h")
    forecast = client.get_forecast_24h()
    print(f"Forecast: {len(forecast)} hourly readings")
    for r in forecast[:4]:
        print(f"  {r.timestamp}: {r.temperature_c:.1f}°C {r.condition}")
