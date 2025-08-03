"""
Fast tests that don't require VecBookIndex or model loading
"""

import pytest
from fastapi.testclient import TestClient

# Import the server app
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evolvattention.server import app, state

client = TestClient(app)


class TestFastEndpoints:
    """Fast tests that don't require model loading"""
    
    def setup_method(self):
        """Reset server state before each test"""
        # Clear server state
        state.barycenter_vector = None
        state.evolution_session = None
        state.vecbook_index = None
    
    def test_health_endpoint(self):
        """Test that health endpoint works"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_static_files(self):
        """Test that static files are served"""
        # Test CSS file
        response = client.get("/index.css")
        assert response.status_code == 200
        
        # Test JS file  
        response = client.get("/index.js")
        assert response.status_code == 200
        
        # Test HTML file
        response = client.get("/")
        assert response.status_code == 200
    
    def test_error_handling_no_barycenter(self):
        """Test error handling when barycenter is not set"""
        # Ensure barycenter is not set
        state.barycenter_vector = None
        
        # Try to get similarities without setting barycenter
        test_strings = ["hello there", "goodbye friend"]
        response = client.post("/api/v1/cosine-similarities", json={"strings": test_strings})
        
        assert response.status_code == 400
        data = response.json()
        assert "No barycenter set" in data["detail"]
    
    def test_error_handling_no_evolution_session(self):
        """Test error handling when evolution session is not active"""
        # Ensure evolution session is not set
        state.evolution_session = None
        
        response = client.post("/api/v1/evolution/step")
        
        assert response.status_code == 400
        data = response.json()
        assert "No active evolution session" in data["detail"]
    
    def test_error_handling_no_attention_barycenter(self):
        """Test error handling when attention analysis is called without barycenter"""
        # Ensure barycenter is not set
        state.barycenter_vector = None
        
        response = client.post("/api/v1/attention/analyze", json={"string": "test string"})
        
        assert response.status_code == 400
        data = response.json()
        assert "No barycenter available" in data["detail"]


if __name__ == "__main__":
    pytest.main([__file__])

# Test Partitioning Guide:
# 
# Fast tests (no model loading, no embeddings):
#   pytest tests/ -v
# 
# Slow tests (with model loading and embeddings):
#   pytest tests_slow/ -v
# 
# All tests:
#   pytest -v
# 
# Skip slow tests during development:
#   pytest tests/ -v
