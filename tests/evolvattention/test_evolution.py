"""
Unit tests for evolutionary algorithm implementation
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

# Import the modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from evolvattention.evolve import Individual, Population, EvolutionaryAlgorithm
from evolvattention.vecx.vecbook_index import VecBookIndex


class TestIndividual:
    """Test Individual class"""
    
    def test_individual_creation(self):
        """Test creating an individual"""
        text = "test string"
        individual = Individual(text=text)
        
        assert individual.text == text
        assert individual.fitness == 0.0
        assert len(individual.attention_scores) == 0
        assert individual.embedding is not None
    
    def test_individual_with_embedding(self):
        """Test creating an individual with embedding"""
        text = "test string"
        embedding = np.array([1.0, 2.0, 3.0])
        individual = Individual(text=text, embedding=embedding)
        
        assert individual.text == text
        assert np.array_equal(individual.embedding, embedding)


class TestPopulation:
    """Test Population class"""
    
    def test_population_creation(self):
        """Test creating a population"""
        individuals = [
            Individual(text="test1", fitness=0.8),
            Individual(text="test2", fitness=0.6),
            Individual(text="test3", fitness=0.9)
        ]
        target_barycenter = np.array([1.0, 2.0, 3.0])
        
        population = Population(individuals=individuals, target_barycenter=target_barycenter)
        
        assert len(population.individuals) == 3
        assert population.generation == 0
        assert population.best_fitness == 0.9
        assert population.average_fitness == pytest.approx(0.766666, rel=1e-5)
        assert np.array_equal(population.target_barycenter, target_barycenter)
    
    def test_population_sort_by_fitness(self):
        """Test sorting population by fitness"""
        individuals = [
            Individual(text="test1", fitness=0.8),
            Individual(text="test2", fitness=0.6),
            Individual(text="test3", fitness=0.9)
        ]
        target_barycenter = np.array([1.0, 2.0, 3.0])
        
        population = Population(individuals=individuals, target_barycenter=target_barycenter)
        population.sort_by_fitness()
        
        assert population.individuals[0].fitness == 0.9
        assert population.individuals[1].fitness == 0.8
        assert population.individuals[2].fitness == 0.6
    
    def test_get_best_individual(self):
        """Test getting best individual"""
        individuals = [
            Individual(text="test1", fitness=0.8),
            Individual(text="test2", fitness=0.6),
            Individual(text="test3", fitness=0.9)
        ]
        target_barycenter = np.array([1.0, 2.0, 3.0])
        
        population = Population(individuals=individuals, target_barycenter=target_barycenter)
        best = population.get_best_individual()
        
        assert best.fitness == 0.9
        assert best.text == "test3"


class TestEvolutionaryAlgorithm:
    """Test EvolutionaryAlgorithm class"""
    
    @pytest.fixture
    def mock_vecbook_index(self):
        """Create a mock VecBookIndex"""
        mock = Mock(spec=VecBookIndex)
        
        # Mock set_target_strings
        mock.set_target_strings.return_value = {
            "status": "success",
            "message": "Target strings set successfully",
            "target_count": 2,
            "barycenter_dimension": 384
        }
        
        # Mock get_target_info
        mock.get_target_info.return_value = {
            "status": "success",
            "target_count": 2,
            "target_strings": ["test1", "test2"],
            "barycenter_dimension": 384
        }
        
        # Mock target_barycenter
        mock.target_barycenter = np.array([1.0, 2.0, 3.0])
        
        # Mock compare_against_barycenter
        mock.compare_against_barycenter.return_value = [
            {
                "text": "test string",
                "embedding": [1.0, 2.0, 3.0],
                "cosine_similarity": "0.850000",
                "index": 0
            }
        ]
        
        return mock
    
    def test_evolutionary_algorithm_initialization(self, mock_vecbook_index):
        """Test initializing evolutionary algorithm"""
        ea = EvolutionaryAlgorithm(mock_vecbook_index)
        
        assert ea.vecbook_index == mock_vecbook_index
        assert ea.population_size == 50
        assert ea.elite_size == 5
        assert ea.tournament_size == 3
        assert ea.crossover_rate == 0.8
        assert ea.mutation_rate == 0.1
        assert ea.population is None
        assert ea.generation == 0
    
    def test_initialize_population_success(self, mock_vecbook_index):
        """Test successful population initialization"""
        ea = EvolutionaryAlgorithm(mock_vecbook_index, population_size=10)
        
        target_strings = ["test1", "test2"]
        result = ea.initialize_population(target_strings)
        
        assert result["status"] == "success"
        assert result["population_size"] == 10
        assert ea.population is not None
        assert len(ea.population.individuals) == 10
        assert ea.generation == 0
    
    def test_initialize_population_empty_targets(self, mock_vecbook_index):
        """Test population initialization with empty target strings"""
        ea = EvolutionaryAlgorithm(mock_vecbook_index)
        
        result = ea.initialize_population([])
        
        assert result["status"] == "error"
        assert "empty" in result["message"].lower()
    
    def test_initialize_population_vecbook_error(self, mock_vecbook_index):
        """Test population initialization when VecBookIndex fails"""
        mock_vecbook_index.set_target_strings.return_value = {
            "status": "error",
            "message": "VecBookIndex error"
        }
        
        ea = EvolutionaryAlgorithm(mock_vecbook_index)
        
        result = ea.initialize_population(["test1", "test2"])
        
        assert result["status"] == "error"
        assert "VecBookIndex error" in result["message"]
    
    def test_evaluate_fitness(self, mock_vecbook_index):
        """Test fitness evaluation"""
        ea = EvolutionaryAlgorithm(mock_vecbook_index)
        individual = Individual(text="test string")
        
        fitness = ea._evaluate_fitness(individual)
        
        assert fitness == 0.85
        mock_vecbook_index.compare_against_barycenter.assert_called_once_with(["test string"])
    
    def test_select_parents(self, mock_vecbook_index):
        """Test parent selection"""
        ea = EvolutionaryAlgorithm(mock_vecbook_index, population_size=10)
        
        # Initialize population first
        ea.initialize_population(["test1", "test2"])
        
        parent1, parent2 = ea._select_parents()
        
        assert parent1 is not None
        assert parent2 is not None
        assert isinstance(parent1, Individual)
        assert isinstance(parent2, Individual)
    
    def test_crossover(self, mock_vecbook_index):
        """Test crossover operation"""
        ea = EvolutionaryAlgorithm(mock_vecbook_index)
        
        parent1 = Individual(text="hello world")
        parent2 = Individual(text="goodbye earth")
        
        offspring = ea._crossover(parent1, parent2)
        
        assert isinstance(offspring, Individual)
        assert offspring.text is not None
        assert len(offspring.text) > 0
    
    def test_mutate(self, mock_vecbook_index):
        """Test mutation operation"""
        ea = EvolutionaryAlgorithm(mock_vecbook_index)
        
        original = Individual(text="hello world")
        mutated = ea._mutate(original)
        
        assert isinstance(mutated, Individual)
        assert mutated.text is not None
    
    def test_evolve_generation_no_population(self, mock_vecbook_index):
        """Test evolving generation without initialized population"""
        ea = EvolutionaryAlgorithm(mock_vecbook_index)
        
        result = ea.evolve_generation()
        
        assert result["status"] == "error"
        assert "initialized" in result["message"].lower()
    
    def test_evolve_generation_success(self, mock_vecbook_index):
        """Test successful generation evolution"""
        ea = EvolutionaryAlgorithm(mock_vecbook_index, population_size=10)
        
        # Initialize population first
        ea.initialize_population(["test1", "test2"])
        
        result = ea.evolve_generation()
        
        assert result["status"] == "success"
        assert result["generation"] == 1
        assert "best_fitness" in result
        assert "average_fitness" in result
    
    def test_get_status_no_population(self, mock_vecbook_index):
        """Test getting status without initialized population"""
        ea = EvolutionaryAlgorithm(mock_vecbook_index)
        
        result = ea.get_status()
        
        assert result["status"] == "error"
        assert "initialized" in result["message"].lower()
    
    def test_get_status_success(self, mock_vecbook_index):
        """Test getting status with initialized population"""
        ea = EvolutionaryAlgorithm(mock_vecbook_index, population_size=10)
        
        # Initialize population first
        ea.initialize_population(["test1", "test2"])
        
        result = ea.get_status()
        
        assert result["status"] == "success"
        assert "current_generation" in result
        assert "best_fitness" in result
        assert "convergence_rate" in result
        assert "best_string" in result
        assert "is_complete" in result
    
    def test_get_population_data_no_population(self, mock_vecbook_index):
        """Test getting population data without initialized population"""
        ea = EvolutionaryAlgorithm(mock_vecbook_index)
        
        result = ea.get_population_data()
        
        assert result == []
    
    def test_get_population_data_success(self, mock_vecbook_index):
        """Test getting population data with initialized population"""
        ea = EvolutionaryAlgorithm(mock_vecbook_index, population_size=5)
        
        # Initialize population first
        ea.initialize_population(["test1", "test2"])
        
        result = ea.get_population_data()
        
        assert len(result) == 5
        for item in result:
            assert "string" in item
            assert "similarity" in item
            assert isinstance(item["string"], str)
            assert isinstance(item["similarity"], float) 