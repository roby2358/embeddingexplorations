"""
Evolutionary Algorithm Implementation
Implements population management and core evolutionary algorithm components
"""

import logging
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Alphabet constants for random string generation
VISIBLE_ASCII_ALPHABET = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
MUTATION_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:"
POPULATION_ALPHABET = " abcdefghijklmnopqrstuvwxyz"

logger = logging.getLogger(__name__)

@dataclass
class Individual:
    """Represents an individual in the evolutionary population"""
    text: str
    embedding: Optional[np.ndarray] = None
    fitness: float = 0.0
    attention_scores: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        # Keep embedding as None if not provided - will be computed when needed
        pass

@dataclass
class Population:
    """Represents a population of individuals"""
    individuals: List[Individual]
    target_barycenter: Optional[np.ndarray]
    generation: int = 0
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    
    def __post_init__(self):
        # Validate target_barycenter
        if self.target_barycenter is not None:
            if hasattr(self.target_barycenter, 'size') and self.target_barycenter.size == 0:
                raise ValueError("target_barycenter cannot be an empty array")
        
        if self.individuals:
            self._update_statistics()
    
    def _update_statistics(self):
        """Update population statistics"""
        fitnesses = [ind.fitness for ind in self.individuals]
        self.best_fitness = max(fitnesses) if fitnesses else 0.0
        self.average_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
    
    def sort_by_fitness(self):
        """Sort individuals by fitness (best first)"""
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
    
    def get_best_individual(self) -> Optional[Individual]:
        """Get the individual with highest fitness"""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda x: x.fitness)

class EvolutionaryAlgorithm:
    """Main evolutionary algorithm implementation"""
    
    def __init__(self, vecbook_index, population_size: int = 50, 
                 elite_size: int = 5, tournament_size: int = 3,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1):
        self.vecbook_index = vecbook_index
        self.population_size = population_size
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # Evolution state
        self.population: Optional[Population] = None
        self.target_barycenter: Optional[np.ndarray] = None
        self.generation = 0
        self.fitness_history: List[float] = []
        
        logger.info(f"Evolutionary algorithm initialized with population_size={population_size}, "
                   f"elite_size={elite_size}, tournament_size={tournament_size}")
    
    def update_population_size(self, new_population_size: int) -> None:
        """Update the population size for the evolutionary algorithm"""
        self.population_size = new_population_size
        logger.info(f"Population size updated to {new_population_size}")
    
    def initialize_population(self, target_strings: List[str], output_length: int = 100) -> Dict[str, Any]:
        """
        Initialize evolutionary algorithm with target strings.
        
        Args:
            target_strings: List of strings to use as targets for evolution
            output_length: Length of strings to generate
        
        Returns:
            Dict with status and population information
        """
        if not target_strings:
            return {
                "status": "error",
                "message": "Target strings list cannot be empty"
            }
        
        try:
            # Store output_length for use in string generation
            self.output_length = output_length
            
            # Set target strings and calculate barycenter using VecBookIndex
            target_result = self.vecbook_index.set_target_strings(target_strings)
            if target_result["status"] != "success":
                return target_result
            
            # Get target barycenter from VecBookIndex
            target_info = self.vecbook_index.get_target_info()
            if target_info["status"] != "success":
                return target_info
            
            # Create initial population from target strings with variations
            individuals = []
            
            for i in range(self.population_size):
                # Generate individual with random strings from visible ASCII
                individual_text = self._generate_initial_variation("")
                
                # Create individual
                individual = Individual(text=individual_text)
                
                # Calculate fitness using VecBookIndex
                fitness_result = self._evaluate_fitness(individual)
                individual.fitness = fitness_result
                
                individuals.append(individual)
            
            # Ensure target_barycenter is properly set
            if self.vecbook_index.target_barycenter is None:
                return {
                    "status": "error",
                    "message": "Target barycenter not properly set"
                }
            
            # Check if target_barycenter is empty
            if hasattr(self.vecbook_index.target_barycenter, 'size') and self.vecbook_index.target_barycenter.size == 0:
                return {
                    "status": "error",
                    "message": "Target barycenter is empty"
                }
            
            # Create population
            self.population = Population(
                individuals=individuals,
                target_barycenter=self.vecbook_index.target_barycenter,
                generation=0
            )
            
            # Sort by fitness (best first)
            self.population.sort_by_fitness()
            
            # Update generation counter
            self.generation = 0
            self.fitness_history = [self.population.best_fitness]
            
            logger.info(f"Population initialized with {len(individuals)} individuals, "
                       f"best fitness: {self.population.best_fitness:.6f}")
            
            return {
                "status": "success",
                "message": f"Population initialized successfully",
                "population_size": len(individuals),
                "best_fitness": self.population.best_fitness,
                "average_fitness": self.population.average_fitness,
                "generation": self.generation
            }
            
        except Exception as e:
            logger.error(f"Error initializing population: {e}")
            return {
                "status": "error",
                "message": f"Failed to initialize population: {str(e)}"
            }
    
    def _generate_initial_variation(self, base_string: str = "") -> str:
        """Generate random string from visible ASCII characters"""
        # Use output_length from the evolution session
        length = getattr(self, 'output_length', 100)  # Default to 100 if not set
        return ''.join(random.choices(POPULATION_ALPHABET, k=length))
    
    def _evaluate_fitness(self, individual: Individual) -> float:
        """Evaluate fitness of an individual using cosine similarity to target barycenter"""
        try:
            # Use VecBookIndex to compare individual against stored barycenter
            comparison_results = self.vecbook_index.compare_against_barycenter([individual.text])
            
            if comparison_results:
                fitness = float(comparison_results[0]["cosine_similarity"])
                return max(0.0, min(1.0, fitness))  # Ensure between 0 and 1
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error evaluating fitness: {e}")
            return 0.0
    
    def _select_parents(self) -> Tuple[Individual, Individual]:
        """Select two parents using tournament selection"""
        # Select first parent
        tournament1 = random.sample(self.population.individuals, self.tournament_size)
        parent1 = max(tournament1, key=lambda x: x.fitness)
        
        # 25% chance that second parent is a random string
        if random.random() < 0.25:
            # Generate a random string as second parent
            random_text = self._generate_initial_variation("")
            parent2 = Individual(text=random_text)
        else:
            # Select second parent using tournament selection (different from first)
            remaining = [ind for ind in self.population.individuals if ind != parent1]
            if len(remaining) >= self.tournament_size:
                tournament2 = random.sample(remaining, self.tournament_size)
            else:
                tournament2 = remaining
            
            parent2 = max(tournament2, key=lambda x: x.fitness) if tournament2 else parent1
        
        return parent1, parent2
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Perform crossover between two parents"""
        # Simple crossover: combine parent strings with 50/50 chance for each character
        # This is the basic implementation as noted in the SPEC
        
        text1, text2 = parent1.text, parent2.text
        max_len = max(len(text1), len(text2))
        
        # Pad shorter string with spaces
        text1 = text1.ljust(max_len)
        text2 = text2.ljust(max_len)
        
        # Create offspring by choosing characters from parents
        offspring_text = ""
        for i in range(max_len):
            if random.random() < 0.5:
                offspring_text += text1[i] if i < len(text1) else " "
            else:
                offspring_text += text2[i] if i < len(text2) else " "
        
        # Clean up the offspring text
        offspring_text = offspring_text.strip()
        if not offspring_text:
            offspring_text = text1.strip() if text1.strip() else text2.strip()
        
        return Individual(text=offspring_text)
    
    def _mutate(self, individual: Individual) -> Individual:
        """Apply mutation to an individual"""
        text = individual.text
        
        # Character-level mutations
        if random.random() < self.mutation_rate:
            # Random character substitution
            if text:
                pos = random.randint(0, len(text) - 1)
                new_char = random.choice(POPULATION_ALPHABET)
                text = text[:pos] + new_char + text[pos+1:]
        
        if random.random() < self.mutation_rate:
            # Random character insertion
            if text:
                pos = random.randint(0, len(text))
                new_char = random.choice(POPULATION_ALPHABET)
                text = text[:pos] + new_char + text[pos:]
        
        if random.random() < self.mutation_rate:
            # Random character deletion
            if len(text) > 1:
                pos = random.randint(0, len(text) - 1)
                text = text[:pos] + text[pos+1:]
        
        return Individual(text=text)
    
    def evolve_generation(self) -> Dict[str, Any]:
        """
        Evolve the population for one generation.
        
        Returns:
            Dict with status and generation statistics
        """
        if not self.population:
            return {
                "status": "error",
                "message": "No population initialized. Call initialize_population first."
            }
        
        try:
            # Create new population
            new_individuals = []
            
            # Elitism: preserve best individuals
            elite_count = min(self.elite_size, len(self.population.individuals))
            new_individuals.extend(self.population.individuals[:elite_count])
            
            # Track existing strings to avoid duplicates
            existing_strings = set(ind.text for ind in new_individuals)
            
            # Generate offspring
            attempts = 0
            max_attempts = self.population_size * 10  # Prevent infinite loops
            
            while len(new_individuals) < self.population_size and attempts < max_attempts:
                attempts += 1
                
                if random.random() < self.crossover_rate:
                    # Crossover
                    parent1, parent2 = self._select_parents()
                    offspring = self._crossover(parent1, parent2)
                else:
                    # Clone a parent
                    parent = random.choice(self.population.individuals)
                    offspring = Individual(text=parent.text)
                
                # Apply mutation
                offspring = self._mutate(offspring)
                
                # Check if this string already exists
                if offspring.text in existing_strings:
                    # Skip this variant and try again with different parents
                    continue
                
                # Evaluate fitness
                offspring.fitness = self._evaluate_fitness(offspring)
                
                # Add to new population and track
                new_individuals.append(offspring)
                existing_strings.add(offspring.text)
            
            # If we couldn't generate enough unique individuals, fill with random variations
            while len(new_individuals) < self.population_size:
                # Generate a completely random string
                random_text = self._generate_initial_variation("")
                
                # Ensure it's unique
                if random_text not in existing_strings:
                    random_individual = Individual(text=random_text)
                    random_individual.fitness = self._evaluate_fitness(random_individual)
                    new_individuals.append(random_individual)
                    existing_strings.add(random_text)
            
            # Update population
            self.population.individuals = new_individuals[:self.population_size]
            self.population.generation += 1
            self.generation = self.population.generation
            
            # Update statistics
            self.population._update_statistics()
            self.population.sort_by_fitness()
            
            # Update fitness history
            self.fitness_history.append(self.population.best_fitness)
            
            logger.info(f"Generation {self.generation} completed. "
                       f"Best fitness: {self.population.best_fitness:.6f}, "
                       f"Average fitness: {self.population.average_fitness:.6f}")
            
            return {
                "status": "success",
                "generation": self.generation,
                "best_fitness": self.population.best_fitness,
                "average_fitness": self.population.average_fitness,
                "population_size": len(self.population.individuals)
            }
            
        except Exception as e:
            logger.error(f"Error during evolution: {e}")
            return {
                "status": "error",
                "message": f"Evolution failed: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the evolutionary algorithm"""
        if not self.population:
            return {
                "status": "error",
                "message": "No population initialized"
            }
        
        # Calculate convergence rate from recent fitness history
        convergence_rate = 0.0
        if len(self.fitness_history) >= 5:
            recent_improvement = self.fitness_history[-1] - self.fitness_history[-5]
            convergence_rate = max(0.0, recent_improvement)
        
        best_individual = self.population.get_best_individual()
        
        return {
            "status": "success",
            "current_generation": self.generation,
            "best_fitness": self.population.best_fitness,
            "average_fitness": self.population.average_fitness,
            "convergence_rate": convergence_rate,
            "best_string": best_individual.text if best_individual else "",
            "is_complete": False,  # Could implement termination conditions here
            "population_size": len(self.population.individuals)
        }
    
    def get_population_data(self) -> List[Dict[str, Any]]:
        """Get population data for API response"""
        if not self.population:
            return []
        
        return [
            {
                "string": ind.text,
                "similarity": ind.fitness
            }
            for ind in self.population.individuals
        ] 