"""
Test server integration with VecBookIndex methods (slow tests with model loading)
"""

import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient

# Import the server app
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evolvattention.server import app, state

client = TestClient(app)


class TestVecBookIndexIntegration:
    """Test that VecBookIndex methods are properly connected to HTTP endpoints"""
    
    def setup_method(self):
        """Reset server state before each test"""
        # Clear server state
        state.barycenter_vector = None
        state.evolution_session = None
        state.vecbook_index = None
    
    def test_barycenter_endpoint(self):
        """Test that /api/v1/barycenter uses VecBookIndex.set_target_strings"""
        test_strings = ["hello world", "goodbye world", "test string"]
        
        response = client.post("/api/v1/barycenter", json={"strings": test_strings})
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    def test_cosine_similarities_endpoint(self):
        """Test that /api/v1/cosine-similarities uses VecBookIndex.compare_against_barycenter"""
        # First set barycenter
        target_strings = ["hello world", "goodbye world"]
        client.post("/api/v1/barycenter", json={"strings": target_strings})
        
        # Then test similarities
        test_strings = ["hello there", "goodbye friend", "unrelated text"]
        response = client.post("/api/v1/cosine-similarities", json={"strings": test_strings})
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "similarities" in data["data"]
        assert len(data["data"]["similarities"]) == len(test_strings)
        
        # Verify similarities are between 0 and 1
        for similarity in data["data"]["similarities"]:
            assert 0.0 <= similarity <= 1.0
    
    def test_evolution_initialize_endpoint(self):
        """Test that /api/v1/evolution/initialize uses VecBookIndex methods"""
        target_strings = ["hello world", "goodbye world"]
        request_data = {
            "target_strings": target_strings,
            "population_size": 5,
            "step_generations": 2,
            "output_length": 50
        }
        
        response = client.post("/api/v1/evolution/initialize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "population" in data["data"]
        assert len(data["data"]["population"]) == 5
        
        # Verify each population member has string and similarity
        for member in data["data"]["population"]:
            assert "string" in member
            assert "similarity" in member
            assert 0.0 <= member["similarity"] <= 1.0
    
    def test_attention_analyze_endpoint(self):
        """Test that /api/v1/attention/analyze uses VecBookIndex.compare_against_barycenter"""
        # First set barycenter
        target_strings = ["hello world", "goodbye world"]
        client.post("/api/v1/barycenter", json={"strings": target_strings})
        
        # Then analyze attention
        test_string = "hello there friend"
        response = client.post("/api/v1/attention/analyze", json={"string": test_string})
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "components" in data["data"]
        assert "overall_score" in data["data"]
        
        # Verify components have expected structure
        for component in data["data"]["components"]:
            assert "text" in component
            assert "score" in component
            assert "position" in component
            assert 0.0 <= component["score"] <= 1.0


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