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
MUTATION_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:"
)
POPULATION_ALPHABET = " abcdefghijklmnopqrstuvwxyz"

logger = logging.getLogger(__name__)

# Bundled Scrabble word list (one lowercase word per line), shipped next to this
# module. Loaded once and shared across all WordCodec instances.
WORDS_PATH = Path(__file__).with_name("words.txt")
_WORD_LIST_CACHE: Optional[List[str]] = None


class CharCodec:
    """Genome = list of single characters; text = those characters joined.

    This is the original behaviour: the alphabet of evolution is individual
    printable characters.
    """

    name = "char"

    def __init__(self, alphabet: str = POPULATION_ALPHABET):
        self.alphabet = alphabet

    def random_unit(self) -> str:
        return random.choice(self.alphabet)

    def random_genome(self, length: int) -> List[str]:
        return random.choices(self.alphabet, k=max(0, length))

    def to_text(self, genome: List[str]) -> str:
        return "".join(genome)


class WordCodec:
    """Genome = list of Scrabble words; text = those words joined by spaces.

    The alphabet of evolution is the bundled Scrabble dictionary (~179k words),
    so every gene is a real word and the evolved output reads as word salad
    rather than character soup. `output_length` counts words.
    """

    name = "word"

    def __init__(self, words_path: Path = WORDS_PATH):
        self.words_path = words_path
        self.vocab = self._load_vocab()

    def _load_vocab(self) -> List[str]:
        global _WORD_LIST_CACHE
        if _WORD_LIST_CACHE is not None:
            return _WORD_LIST_CACHE
        with open(self.words_path, "r", encoding="utf-8") as fh:
            words = [w.strip() for w in fh if w.strip()]
        if not words:
            raise RuntimeError(f"Scrabble word list at {self.words_path} is empty")
        _WORD_LIST_CACHE = words
        logger.info(f"Word codec: loaded {len(words)} words from {self.words_path}")
        return words

    def random_unit(self) -> str:
        return random.choice(self.vocab)

    def random_genome(self, length: int) -> List[str]:
        return [self.random_unit() for _ in range(max(0, length))]

    def to_text(self, genome: List[str]) -> str:
        return " ".join(genome)


@dataclass
class Individual:
    """Represents an individual in the evolutionary population"""

    text: str
    embedding: Optional[np.ndarray] = None
    fitness: float = 0.0
    attention_scores: List[float] = field(default_factory=list)
    genome: list = field(default_factory=list)  # units: chars or words

    def __post_init__(self) -> None:
        # When constructed directly from text (char mode / tests), the genome
        # is just the character sequence. Word-mode individuals are always
        # built with an explicit genome via EvolutionaryAlgorithm._new_individual.
        if not self.genome:
            self.genome = list(self.text)


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
            if (
                hasattr(self.target_barycenter, "size")
                and self.target_barycenter.size == 0
            ):
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

    def __init__(
        self,
        vecbook_index,
        population_size: int = 50,
        elite_size: int = 5,
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        genome_mode: str = "word",
    ):
        self.vecbook_index = vecbook_index
        self.population_size = population_size
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        # Genome representation: "word" (Scrabble words) or "char" (per-character)
        self.genome_mode = genome_mode
        self.codec = self._build_codec(genome_mode)

        # Evolution state
        self.population: Optional[Population] = None
        self.target_barycenter: Optional[np.ndarray] = None
        self.generation = 0
        self.fitness_history: List[float] = []

        logger.info(
            f"Evolutionary algorithm initialized with population_size={population_size}, "
            f"elite_size={elite_size}, tournament_size={tournament_size}, "
            f"genome_mode={genome_mode}"
        )

    def update_population_size(self, new_population_size: int) -> None:
        """Update the population size for the evolutionary algorithm"""
        self.population_size = new_population_size
        logger.info(f"Population size updated to {new_population_size}")

    def _build_codec(self, mode: str) -> Any:
        """Construct the genome codec for the requested mode."""
        if mode == "char":
            return CharCodec()
        if mode == "word":
            return WordCodec()
        raise ValueError(f"Unknown genome_mode: {mode!r} (expected 'char' or 'word')")

    def set_genome_mode(self, mode: str) -> None:
        """Switch the genome representation (rebuilds the codec)."""
        self.codec = self._build_codec(mode)
        self.genome_mode = mode
        logger.info(f"Genome mode set to '{mode}'")

    def _new_individual(self, genome: list) -> Individual:
        """Build an Individual from a genome (list of units), decoding its text."""
        genome = list(genome)
        return Individual(text=self.codec.to_text(genome), genome=genome)

    def _random_individual(self) -> Individual:
        """Build a fresh random Individual using the active codec."""
        length = getattr(self, "output_length", 100)
        return self._new_individual(self.codec.random_genome(length))

    def initialize_population(
        self, target_strings: List[str], output_length: int = 100
    ) -> Dict[str, Any]:
        """
        Initialize evolutionary algorithm with target strings.

        Args:
            target_strings: List of strings to use as targets for evolution
            output_length: Length of strings to generate

        Returns:
            Dict with status and population information
        """
        if not target_strings:
            return {"status": "error", "message": "Target strings list cannot be empty"}

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
                # Generate individual with a random genome using the active codec
                individual = self._random_individual()

                # Calculate fitness using VecBookIndex
                fitness_result = self._evaluate_fitness(individual)
                individual.fitness = fitness_result

                individuals.append(individual)

            # Ensure target_barycenter is properly set
            if self.vecbook_index.target_barycenter is None:
                return {
                    "status": "error",
                    "message": "Target barycenter not properly set",
                }

            # Check if target_barycenter is empty
            if (
                hasattr(self.vecbook_index.target_barycenter, "size")
                and self.vecbook_index.target_barycenter.size == 0
            ):
                return {"status": "error", "message": "Target barycenter is empty"}

            # Create population
            self.population = Population(
                individuals=individuals,
                target_barycenter=self.vecbook_index.target_barycenter,
                generation=0,
            )

            # Sort by fitness (best first)
            self.population.sort_by_fitness()

            # Update generation counter
            self.generation = 0
            self.fitness_history = [self.population.best_fitness]

            logger.info(
                f"Population initialized with {len(individuals)} individuals, "
                f"best fitness: {self.population.best_fitness:.6f}"
            )

            return {
                "status": "success",
                "message": f"Population initialized successfully",
                "population_size": len(individuals),
                "best_fitness": self.population.best_fitness,
                "average_fitness": self.population.average_fitness,
                "generation": self.generation,
            }

        except Exception as e:
            logger.error(f"Error initializing population: {e}")
            return {
                "status": "error",
                "message": f"Failed to initialize population: {str(e)}",
            }

    def _generate_initial_variation(self, base_string: str = "") -> str:
        """Generate random string from visible ASCII characters"""
        # Use output_length from the evolution session
        length = getattr(self, "output_length", 100)  # Default to 100 if not set
        return "".join(random.choices(POPULATION_ALPHABET, k=length))

    def _evaluate_fitness(self, individual: Individual) -> float:
        """Evaluate fitness of an individual using cosine similarity to target barycenter"""
        try:
            # Use VecBookIndex to compare individual against stored barycenter
            comparison_results = self.vecbook_index.compare_against_barycenter(
                [individual.text]
            )

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

        # 25% chance that second parent is a fresh random individual
        if random.random() < 0.25:
            parent2 = self._random_individual()
        else:
            # Select second parent using tournament selection (different from first)
            remaining = [ind for ind in self.population.individuals if ind != parent1]
            if len(remaining) >= self.tournament_size:
                tournament2 = random.sample(remaining, self.tournament_size)
            else:
                tournament2 = remaining

            parent2 = (
                max(tournament2, key=lambda x: x.fitness) if tournament2 else parent1
            )

        return parent1, parent2

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Perform uniform crossover over genome units (chars or tokens).

        For each position, take the unit from one parent or the other with 50/50
        chance. Works identically whether a unit is a character or a token id.
        """
        g1, g2 = parent1.genome, parent2.genome
        max_len = max(len(g1), len(g2))

        offspring_genome: list = []
        for i in range(max_len):
            source = g1 if random.random() < 0.5 else g2
            if i < len(source):
                offspring_genome.append(source[i])

        if not offspring_genome:
            offspring_genome = list(g1) if g1 else list(g2)

        return self._new_individual(offspring_genome)

    def _mutate(self, individual: Individual) -> Individual:
        """Apply substitution / insertion / deletion to genome units."""
        genome = list(individual.genome)

        # Substitution
        if random.random() < self.mutation_rate and genome:
            pos = random.randrange(len(genome))
            genome[pos] = self.codec.random_unit()

        # Insertion
        if random.random() < self.mutation_rate:
            pos = random.randint(0, len(genome))
            genome.insert(pos, self.codec.random_unit())

        # Deletion
        if random.random() < self.mutation_rate and len(genome) > 1:
            pos = random.randrange(len(genome))
            del genome[pos]

        return self._new_individual(genome)

    def evolve_generation(self) -> Dict[str, Any]:
        """
        Evolve the population for one generation.

        Returns:
            Dict with status and generation statistics
        """
        if not self.population:
            return {
                "status": "error",
                "message": "No population initialized. Call initialize_population first.",
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

            while (
                len(new_individuals) < self.population_size and attempts < max_attempts
            ):
                attempts += 1

                if random.random() < self.crossover_rate:
                    # Crossover
                    parent1, parent2 = self._select_parents()
                    offspring = self._crossover(parent1, parent2)
                else:
                    # Clone a parent (copy its genome so later mutation is independent)
                    parent = random.choice(self.population.individuals)
                    offspring = self._new_individual(parent.genome)

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

            # If we couldn't generate enough unique individuals, fill with random ones
            while len(new_individuals) < self.population_size:
                random_individual = self._random_individual()

                # Ensure it's unique
                if random_individual.text not in existing_strings:
                    random_individual.fitness = self._evaluate_fitness(
                        random_individual
                    )
                    new_individuals.append(random_individual)
                    existing_strings.add(random_individual.text)

            # Update population
            self.population.individuals = new_individuals[: self.population_size]
            self.population.generation += 1
            self.generation = self.population.generation

            # Update statistics
            self.population._update_statistics()
            self.population.sort_by_fitness()

            # Update fitness history
            self.fitness_history.append(self.population.best_fitness)

            logger.info(
                f"Generation {self.generation} completed. "
                f"Best fitness: {self.population.best_fitness:.6f}, "
                f"Average fitness: {self.population.average_fitness:.6f}"
            )

            return {
                "status": "success",
                "generation": self.generation,
                "best_fitness": self.population.best_fitness,
                "average_fitness": self.population.average_fitness,
                "population_size": len(self.population.individuals),
            }

        except Exception as e:
            logger.error(f"Error during evolution: {e}")
            return {"status": "error", "message": f"Evolution failed: {str(e)}"}

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the evolutionary algorithm"""
        if not self.population:
            return {"status": "error", "message": "No population initialized"}

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
            "population_size": len(self.population.individuals),
        }

    def get_population_data(self) -> List[Dict[str, Any]]:
        """Get population data for API response"""
        if not self.population:
            return []

        return [
            {"string": ind.text, "similarity": ind.fitness}
            for ind in self.population.individuals
        ]
