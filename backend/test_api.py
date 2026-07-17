"""
Validation tests for Smart Traffic AI - Bengaluru API Services
Tests all FastAPIs using TestClient
"""

import sys
from fastapi.testclient import TestClient

# Add local path to import main
sys.path.append(".")
from main import app, GATLSTMPredictor

client = TestClient(app)

def test_health():
    print("Testing /health ...")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("[OK] /health passed")

def test_junctions_live():
    print("Testing /api/junctions/live ...")
    response = client.get("/api/junctions/live")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 15
    assert "name" in data[0]
    assert "congestion" in data[0]
    print("[OK] /api/junctions/live passed")

def test_stats_summary():
    print("Testing /api/stats/summary ...")
    response = client.get("/api/stats/summary")
    assert response.status_code == 200
    data = response.json()
    assert "avg_speed_kmph" in data
    assert "junctions_monitored" in data
    assert "high_congestion_count" in data
    print("[OK] /api/stats/summary passed")

def test_places_lookup():
    print("Testing /api/places/lookup ...")
    # Coordinates of MG Road area
    response = client.get("/api/places/lookup?lat=12.9756&lng=77.6068")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "display_name" in data
    assert data["lat"] == 12.9756
    print("[OK] /api/places/lookup passed")

def test_flood_risk():
    print("Testing /api/flood/risk ...")
    # Coordinates near Silk Board
    response = client.get("/api/flood/risk?lat=12.9172&lng=77.6228")
    assert response.status_code == 200
    data = response.json()
    assert "elevation_m" in data
    assert "precipitation_mm_hr" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["Low", "Moderate", "High"]
    print("[OK] /api/flood/risk passed")

def test_predict_traffic():
    print("Testing /api/predict/traffic ...")
    # Dynamic point coordinate payload
    payload = {"lat": 12.9569, "lng": 77.7011, "junction_id": "J02"}
    response = client.post("/api/predict/traffic", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "forecast_series" in data
    assert len(data["forecast_series"]) == 7
    assert data["forecast_series"][0]["time_offset"] == "now"
    print("[OK] /api/predict/traffic passed")

def test_signals_adaptive():
    print("Testing /api/signals/adaptive ...")
    # Webster split timing check
    response = client.get("/api/signals/adaptive?junction_id=J01&yolo_count=24")
    assert response.status_code == 200
    data = response.json()
    assert "optimum_cycle_time_sec" in data
    assert data["yolo_active"] is True
    assert data["yolo_count"] == 24
    assert len(data["phases"]) == 4
    print("[OK] /api/signals/adaptive passed")

def test_emergency_route_junctions():
    print("Testing /api/emergency/route (Junction to Junction) ...")
    payload = {"origin": "J01", "destination": "J09", "alternatives": 2}
    response = client.post("/api/emergency/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "primary" in data
    assert "routing_engine" in data
    assert "distance_km" in data["primary"]
    assert "eta_minutes" in data["primary"]
    assert "geometry" in data["primary"]
    print("[OK] /api/emergency/route (junctions) passed")

def test_emergency_route_coordinates():
    print("Testing /api/emergency/route (Custom Coordinates) ...")
    # Silk Board area to MG road area
    payload = {"origin": "12.9172,77.6228", "destination": "12.9756,77.6068", "alternatives": 2}
    response = client.post("/api/emergency/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "primary" in data
    assert "routing_engine" in data
    assert "geometry" in data["primary"]
    print("[OK] /api/emergency/route (coordinates) passed")

def test_model_metrics():
    print("Testing /api/model/metrics ...")
    response = client.get("/api/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "gat_lstm" in data
    assert "yolov8" in data
    assert "flood_classifier" in data
    assert "history_feed" in data
    assert "loss_curves" in data
    assert data["gat_lstm"]["mean_absolute_error"] > 0
    assert len(data["history_feed"]) > 0
    print("[OK] /api/model/metrics passed")

if __name__ == "__main__":
    print("Starting smart-traffic API validation suite...")
    test_health()
    test_junctions_live()
    test_stats_summary()
    test_places_lookup()
    test_flood_risk()
    test_predict_traffic()
    test_signals_adaptive()
    test_emergency_route_junctions()
    test_emergency_route_coordinates()
    test_model_metrics()
    print("\nAll 10 integration validation tests completed successfully! No errors found.")
