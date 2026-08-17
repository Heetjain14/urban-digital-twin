"""
environment.py

Global environment state:
- Weather
- Time-of-day
- Traffic lights
- Active scenarios

OpenWeather integration:
- Fetches real current weather for Mumbai
- Uses real temperature, rain and weather condition
- Falls back to synthetic weather if API is unavailable
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import requests
from dotenv import load_dotenv

from src.data_ingestion.schemas import WeatherCondition, WeatherState


# ---------------------------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------------------------

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class CityEnvironment:
    """
    Mutable global state updated every simulation tick.

    All agents read environmental information from this object.

    Weather can come from:
        1. OpenWeather API
        2. Synthetic fallback model

    RL agent can modify traffic-light phases.
    """

    tick: int = 0
    seed: int = 42

    # -----------------------------------------------------------------------
    # Weather state
    # -----------------------------------------------------------------------

    weather_condition: WeatherCondition = WeatherCondition.CLEAR

    temperature_c: float = 22.0

    precipitation_mm_h: float = 0.0

    # Real weather information
    weather_source: str = "synthetic"

    weather_description: str = ""

    humidity_pct: float = 0.0

    wind_speed_mps: float = 0.0

    # -----------------------------------------------------------------------
    # OpenWeather configuration
    # -----------------------------------------------------------------------

    openweather_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENWEATHER_API_KEY")
    )

    weather_city: str = "Mumbai,IN"

    weather_api_enabled: bool = True

    # Fetch real weather every N simulation ticks.
    #
    # Your simulation uses:
    # 1440 ticks = 1 simulated day
    #
    # 60 ticks = 1 simulated hour
    #
    # We therefore fetch OpenWeather once every 60 ticks.
    weather_api_interval: int = 60

    # -----------------------------------------------------------------------
    # Traffic light control
    # -----------------------------------------------------------------------

    rl_control_active: bool = False

    traffic_light_phases: Dict[int, str] = field(
        default_factory=dict
    )

    traffic_light_timers: Dict[int, int] = field(
        default_factory=dict
    )

    # -----------------------------------------------------------------------
    # Active scenario
    # -----------------------------------------------------------------------

    scenario_active: Optional[str] = None

    scenario_params: Dict[str, Any] = field(
        default_factory=dict
    )

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def __post_init__(self):

        # Random number generator used by the synthetic fallback model.
        self.rng = np.random.default_rng(self.seed)

        # Synthetic weather transition interval.
        self._weather_change_interval = 60

        # Track whether API was successfully contacted.
        self._weather_api_available = bool(
            self.openweather_api_key
        )

        if self.openweather_api_key:

            logger.info(
                "OpenWeather API key detected. "
                "Real weather integration enabled."
            )

        else:

            logger.warning(
                "OPENWEATHER_API_KEY not found. "
                "Using synthetic weather."
            )

    # =======================================================================
    # TIME HELPERS
    # =======================================================================

    @property
    def hour_of_day(self) -> float:
        """
        Current simulated hour.

        0.0 → midnight
        12.0 → noon
        24.0 → next midnight
        """

        return (self.tick % 1440) / 60.0

    @property
    def day_of_week(self) -> int:
        """
        0 = Monday
        6 = Sunday
        """

        return (self.tick // 1440) % 7

    @property
    def is_weekend(self) -> bool:

        return self.day_of_week >= 5

    @property
    def is_rush_hour(self) -> bool:

        h = self.hour_of_day

        return (
            (7.0 <= h <= 9.5)
            or
            (16.5 <= h <= 19.0)
        )

    @property
    def sim_datetime_str(self) -> str:

        day = self.tick // 1440

        hour = int(self.hour_of_day)

        minute = int(
            (self.hour_of_day - hour) * 60
        )

        return (
            f"Day {day + 1:02d}  "
            f"{hour:02d}:{minute:02d}"
        )

    # =======================================================================
    # WEATHER EFFECTS
    # =======================================================================

    @property
    def speed_multiplier(self) -> float:

        if self.weather_condition == WeatherCondition.HEAVY_RAIN:

            return 0.50

        elif self.weather_condition == WeatherCondition.LIGHT_RAIN:

            return 0.70

        elif self.weather_condition == WeatherCondition.HOT:

            return 0.90

        return 1.00

    @property
    def energy_multiplier(self) -> float:

        if self.temperature_c > 32:

            # Increased air-conditioning demand
            return 1.25

        elif self.temperature_c < 10:

            # Increased heating demand
            return 1.20

        elif self.weather_condition in (
            WeatherCondition.LIGHT_RAIN,
            WeatherCondition.HEAVY_RAIN,
        ):

            return 1.10

        return 1.00

    @property
    def pedestrian_rate_multiplier(self) -> float:

        if self.weather_condition == WeatherCondition.HEAVY_RAIN:

            return 0.40

        elif self.weather_condition == WeatherCondition.LIGHT_RAIN:

            return 0.70

        return 1.00

    # =======================================================================
    # WEATHER STATE FOR DASHBOARD / OTHER COMPONENTS
    # =======================================================================

    def get_weather_state(self) -> WeatherState:

        return WeatherState(
            condition=self.weather_condition,
            temperature_c=self.temperature_c,
            precipitation_mm_h=self.precipitation_mm_h,
            speed_multiplier=self.speed_multiplier,
            energy_multiplier=self.energy_multiplier,
        )

    # =======================================================================
    # OPENWEATHER API
    # =======================================================================

    def fetch_real_weather(self) -> bool:
        """
        Fetch current weather from OpenWeather.

        Returns:
            True  -> API request successful
            False -> API request failed
        """

        if not self.weather_api_enabled:

            return False

        if not self.openweather_api_key:

            return False

        url = (
            "https://api.openweathermap.org/data/2.5/weather"
        )

        params = {
            "q": self.weather_city,
            "appid": self.openweather_api_key,
            "units": "metric",
        }

        try:

            response = requests.get(
                url,
                params=params,
                timeout=10,
            )

            response.raise_for_status()

            data = response.json()

            # ---------------------------------------------------------------
            # Temperature
            # ---------------------------------------------------------------

            self.temperature_c = float(
                data["main"]["temp"]
            )

            # ---------------------------------------------------------------
            # Humidity
            # ---------------------------------------------------------------

            self.humidity_pct = float(
                data["main"].get("humidity", 0)
            )

            # ---------------------------------------------------------------
            # Wind
            # ---------------------------------------------------------------

            self.wind_speed_mps = float(
                data.get("wind", {}).get("speed", 0)
            )

            # ---------------------------------------------------------------
            # Weather description
            # ---------------------------------------------------------------

            weather_data = data.get("weather", [{}])[0]

            main_condition = weather_data.get(
                "main",
                "Clear"
            )

            self.weather_description = weather_data.get(
                "description",
                ""
            )

            # ---------------------------------------------------------------
            # Rain
            #
            # OpenWeather may return:
            #
            # rain:
            #     1h: 2.5
            #
            # ---------------------------------------------------------------

            rain_data = data.get("rain", {})

            self.precipitation_mm_h = float(
                rain_data.get("1h", 0.0)
            )

            # ---------------------------------------------------------------
            # Convert OpenWeather condition → project condition
            # ---------------------------------------------------------------

            self.weather_condition = (
                self._map_openweather_condition(
                    main_condition,
                    self.precipitation_mm_h,
                )
            )

            self.weather_source = "openweather"

            self._weather_api_available = True

            logger.info(
                "OpenWeather updated: "
                f"{self.weather_city} | "
                f"{self.temperature_c:.1f}°C | "
                f"{main_condition} | "
                f"rain={self.precipitation_mm_h:.1f} mm/h"
            )

            return True

        except requests.RequestException as e:

            logger.warning(
                f"OpenWeather request failed: {e}"
            )

        except (KeyError, TypeError, ValueError) as e:

            logger.warning(
                f"Invalid OpenWeather response: {e}"
            )

        except Exception as e:

            logger.warning(
                f"Unexpected OpenWeather error: {e}"
            )

        self._weather_api_available = False

        return False

    # =======================================================================
    # OPENWEATHER → PROJECT WEATHER CONDITION
    # =======================================================================

    def _map_openweather_condition(
        self,
        condition: str,
        rain_mm_h: float,
    ) -> WeatherCondition:
        """
        Convert OpenWeather's weather categories
        into the WeatherCondition enum used by the project.
        """

        condition = condition.lower()

        # ---------------------------------------------------------------
        # Rain
        # ---------------------------------------------------------------

        if rain_mm_h >= 10:

            return WeatherCondition.HEAVY_RAIN

        if rain_mm_h > 0:

            return WeatherCondition.LIGHT_RAIN

        # ---------------------------------------------------------------
        # OpenWeather rain categories
        # ---------------------------------------------------------------

        if condition in (
            "rain",
            "drizzle",
        ):

            return WeatherCondition.LIGHT_RAIN

        # ---------------------------------------------------------------
        # Thunderstorm
        # ---------------------------------------------------------------

        if condition == "thunderstorm":

            return WeatherCondition.HEAVY_RAIN

        # ---------------------------------------------------------------
        # Very hot weather
        # ---------------------------------------------------------------

        if self.temperature_c >= 35:

            return WeatherCondition.HOT

        # ---------------------------------------------------------------
        # Clear / clouds / normal weather
        # ---------------------------------------------------------------

        return WeatherCondition.CLEAR

    # =======================================================================
    # SIMULATION UPDATE
    # =======================================================================

    def update(self, tick: int):

        self.tick = tick

        # ---------------------------------------------------------------
        # Scenario rain takes priority
        # ---------------------------------------------------------------

        if self.scenario_active == "rain":

            self._update_scenario_weather()

        # ---------------------------------------------------------------
        # Otherwise use OpenWeather
        # ---------------------------------------------------------------

        elif (
            self.weather_api_enabled
            and self.openweather_api_key
            and tick % self.weather_api_interval == 0
        ):

            success = self.fetch_real_weather()

            # If API fails, temporarily use synthetic weather.
            if not success:

                self._update_synthetic_weather()

        # ---------------------------------------------------------------
        # No API key → synthetic weather
        # ---------------------------------------------------------------

        elif not self.openweather_api_key:

            if tick % self._weather_change_interval == 0:

                self._update_synthetic_weather()

        # ---------------------------------------------------------------
        # Traffic light countdown
        # ---------------------------------------------------------------

        for iid in list(
            self.traffic_light_timers.keys()
        ):

            self.traffic_light_timers[iid] = max(
                0,
                self.traffic_light_timers[iid] - 1
            )

            if self.traffic_light_timers[iid] == 0:

                self._toggle_light(iid)

    # =======================================================================
    # SYNTHETIC WEATHER FALLBACK
    # =======================================================================

    def _update_synthetic_weather(self):
        """
        Original stochastic weather model.

        Used when:
        - API key is missing
        - OpenWeather is unavailable
        - API request fails
        """

        self.weather_source = "synthetic"

        if self.scenario_active == "rain":

            self._update_scenario_weather()

            return

        r = self.rng.random()

        # ---------------------------------------------------------------
        # Clear → rain
        # ---------------------------------------------------------------

        if self.weather_condition == WeatherCondition.CLEAR:

            if r < 0.05:

                self.weather_condition = (
                    WeatherCondition.LIGHT_RAIN
                )

                self.precipitation_mm_h = (
                    self.rng.uniform(1, 5)
                )

        # ---------------------------------------------------------------
        # Light rain
        # ---------------------------------------------------------------

        elif self.weather_condition == WeatherCondition.LIGHT_RAIN:

            if r < 0.30:

                self.weather_condition = (
                    WeatherCondition.CLEAR
                )

                self.precipitation_mm_h = 0.0

            elif r < 0.40:

                self.weather_condition = (
                    WeatherCondition.HEAVY_RAIN
                )

                self.precipitation_mm_h = (
                    self.rng.uniform(10, 25)
                )

        # ---------------------------------------------------------------
        # Heavy rain
        # ---------------------------------------------------------------

        elif self.weather_condition == WeatherCondition.HEAVY_RAIN:

            if r < 0.25:

                self.weather_condition = (
                    WeatherCondition.LIGHT_RAIN
                )

                self.precipitation_mm_h = (
                    self.rng.uniform(1, 5)
                )

        # ---------------------------------------------------------------
        # Temperature drift
        # ---------------------------------------------------------------

        self.temperature_c += self.rng.normal(
            0,
            0.5
        )

        self.temperature_c = float(
            np.clip(
                self.temperature_c,
                5,
                45
            )
        )

    # =======================================================================
    # SCENARIO WEATHER
    # =======================================================================

    def _update_scenario_weather(self):

        intensity = self.scenario_params.get(
            "intensity",
            "heavy"
        )

        if intensity == "heavy":

            self.weather_condition = (
                WeatherCondition.HEAVY_RAIN
            )

            self.precipitation_mm_h = 15.0

        else:

            self.weather_condition = (
                WeatherCondition.LIGHT_RAIN
            )

            self.precipitation_mm_h = 4.0

        self.weather_source = "scenario"

    # =======================================================================
    # TRAFFIC LIGHTS
    # =======================================================================

    def initialize_traffic_lights(
        self,
        intersection_ids: List[int],
        default_green_ticks: int = 30,
    ):
        """
        Set up fixed-cycle traffic lights.
        """

        for i, iid in enumerate(intersection_ids):

            # Stagger phases so all intersections
            # don't switch simultaneously.

            phase = (
                "NS_GREEN"
                if i % 2 == 0
                else "EW_GREEN"
            )

            self.traffic_light_phases[iid] = phase

            self.traffic_light_timers[iid] = (
                default_green_ticks
                + (i * 5 % 15)
            )

    def _toggle_light(
        self,
        iid: int,
        green_ticks: int = 30,
    ):

        cur = self.traffic_light_phases.get(
            iid,
            "NS_GREEN"
        )

        self.traffic_light_phases[iid] = (
            "EW_GREEN"
            if cur == "NS_GREEN"
            else "NS_GREEN"
        )

        self.traffic_light_timers[iid] = (
            green_ticks
        )

    def set_rl_phase(
        self,
        iid: int,
        phase: str,
        duration_ticks: int,
    ):
        """
        RL agent sets a specific traffic-light phase.
        """

        self.traffic_light_phases[iid] = phase

        self.traffic_light_timers[iid] = (
            duration_ticks
        )

    # =======================================================================
    # SCENARIO CONTROL
    # =======================================================================

    def apply_scenario(
        self,
        scenario_type: str,
        params: Dict[str, Any],
    ):

        self.scenario_active = scenario_type

        self.scenario_params = params

    def clear_scenario(self):

        self.scenario_active = None

        self.scenario_params = {}

        self.weather_condition = (
            WeatherCondition.CLEAR
        )

        self.precipitation_mm_h = 0.0

        self.weather_source = "synthetic"