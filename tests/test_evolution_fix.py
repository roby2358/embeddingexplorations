"""
Test to verify that the empty array error is fixed
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evolvattention.evolve.evolution import Individual, Population, EvolutionaryAlgorithm
from evolvattention.vecx.vecbook_index import VecBookIndex
from pathlib import Path


class TestEvolutionFix:
    """Test that the empty array error is fixed"""
    
    def setup_method(self):
        """Set up test environment"""
        # Create a temporary data directory
        self.data_path = Path("test_data")
        self.data_path.mkdir(exist_ok=True)
        
        # Create VecBookIndex
        self.vecbook_index = VecBookIndex(self.data_path)
        
        # Create EvolutionaryAlgorithm
        self.ea = EvolutionaryAlgorithm(self.vecbook_index)
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        if self.data_path.exists():
            shutil.rmtree(self.data_path)
    
    def test_individual_creation_no_empty_array(self):
        """Test that Individual creation doesn't create empty arrays"""
        individual = Individual(text="test string")
        
        # Check that embedding is None, not an empty array
        assert individual.embedding is None
    
    def test_population_initialization_with_valid_barycenter(self):
        """Test that population initialization works with valid barycenter"""
        # Set up target strings
        target_strings = ["hello world", "goodbye friend"]
        
        # Initialize population
        result = self.ea.initialize_population(target_strings)
        
        # Should succeed
        assert result["status"] == "success"
        assert self.ea.population is not None
        assert len(self.ea.population.individuals) > 0
    
    def test_evolution_step_no_empty_array_error(self):
        """Test that evolution step doesn't throw empty array error"""
        # Set up target strings
        target_strings = ["hello world", "goodbye friend"]
        
        # Initialize population
        init_result = self.ea.initialize_population(target_strings)
        assert init_result["status"] == "success"
        
        # Run one generation
        step_result = self.ea.evolve_generation()
        
        # Should succeed without empty array error
        assert step_result["status"] == "success"
        assert step_result["generation"] == 1
    
    def test_population_validation_prevents_empty_arrays(self):
        """Test that population validation prevents empty arrays"""
        # Test that creating a population with empty barycenter raises error
        individuals = [Individual(text="test")]
        empty_barycenter = np.array([])
        
        with pytest.raises(ValueError, match="target_barycenter cannot be an empty array"):
            Population(individuals=individuals, target_barycenter=empty_barycenter)
    
    def test_vecbook_index_handles_empty_embeddings(self):
        """Test that VecBookIndex properly handles empty embeddings"""
        # Test with empty target strings (should fail gracefully)
        result = self.vecbook_index.set_target_strings([])
        assert result["status"] == "error"
        assert "empty" in result["message"].lower()


if __name__ == "__main__":
    pytest.main([__file__]) 