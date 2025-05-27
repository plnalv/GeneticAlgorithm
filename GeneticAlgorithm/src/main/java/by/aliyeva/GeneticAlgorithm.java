package by.aliyeva;

import java.util.Arrays;
import java.util.Random;

public class GeneticAlgorithm {
    private static final int POPULATION_SIZE = 100; // Размер популяции
    private static final int MAX_GENERATIONS = 1000; // Максимальное число поколений
    private static final double MUTATION_PROBABILITY = 0.1; // Вероятность мутации
    private static final int TOURNAMENT_SIZE = 5; // Размер турнира для селекции
    private static final double EXTINCTION_THRESHOLD = 0.01; // Порог для вымирания
    private static final int STAGNATION_LIMIT = 50; // Лимит стагнации до вымирания
    private static final double EXTINCTION_RATE = 0.2; // Доля популяции для вымирания
    private static final int RANGE_MIN = -200; // Минимальное значение генов
    private static final int RANGE_MAX = 200; // Максимальное значение генов
    private static final double[] TARGETS = {-5, 49, -50}; // Целевые значения уравнений

    private static class Individual {
        int[] genes = new int[5]; // x1, x2, x3, x4, x5
        double fitness;

        // начальная популяция

        Individual(Random rand) {
            for (int i = 0; i < 5; i++) {
                genes[i] = rand.nextInt(RANGE_MAX - RANGE_MIN + 1) + RANGE_MIN;
            }
            fitness = calculateFitness();
        }

        // функция приспособленности

        double calculateFitness() {
            double[] results = new double[3];
            // Equation 1: x2^2 * x3 * x4^2 * x5 + x2^2 * x3^2 * x4^2 * x5 + x1^2 * x2 * x3 + x2^2 * x3^2 * x4^2 * x5^2 + x4 * x5 = -5
            results[0] = Math.pow(genes[1], 2) * genes[2] * Math.pow(genes[3], 2) * genes[4] +
                    Math.pow(genes[1], 2) * Math.pow(genes[2], 2) * Math.pow(genes[3], 2) * genes[4] +
                    Math.pow(genes[0], 2) * genes[1] * genes[2] +
                    Math.pow(genes[1], 2) * Math.pow(genes[2], 2) * Math.pow(genes[3], 2) * Math.pow(genes[4], 2) +
                    genes[3] * genes[4];
            // Equation 2: x1 * x2 * x3^2 * x4^2 + x2 * x3^2 * x4 + x4 + x1^2 * x2 * x4^2 * x5^2 = 49
            results[1] = genes[0] * genes[1] * Math.pow(genes[2], 2) * Math.pow(genes[3], 2) +
                    genes[1] * Math.pow(genes[2], 2) * genes[3] +
                    genes[3] +
                    Math.pow(genes[0], 2) * genes[1] * Math.pow(genes[3], 2) * Math.pow(genes[4], 2);
            // Equation 3: x3 * x4 * x5^2 + x5 + x1^2 * x2^2 * x3^2 * x4 + x1^2 * x3^2 * x4 + x3^2 * x4 = -50
            results[2] = genes[2] * genes[3] * Math.pow(genes[4], 2) +
                    genes[4] +
                    Math.pow(genes[0], 2) * Math.pow(genes[1], 2) * Math.pow(genes[2], 2) * genes[3] +
                    Math.pow(genes[0], 2) * Math.pow(genes[2], 2) * genes[3] +
                    Math.pow(genes[2], 2) * genes[3];

            double totalError = 0;
            for (int i = 0; i < 3; i++) {
                totalError += Math.abs(results[i] - TARGETS[i]);
            }
            return -totalError / 3.0; // отриц средняя абсолютная ошибка
        }
    }

    // селекция - турнирная

    private static Individual tournamentSelection(Individual[] population, Random rand) {
        Individual best = population[rand.nextInt(POPULATION_SIZE)];
        for (int i = 1; i < TOURNAMENT_SIZE; i++) {
            Individual contender = population[rand.nextInt(POPULATION_SIZE)];
            if (contender.fitness > best.fitness) {
                best = contender;
            }
        }
        return best;
    }

    // скрещивание - пропорциональное вероятностное

    private static Individual crossover(Individual parent1, Individual parent2, Random rand) {
        Individual offspring = new Individual(new Random());
        double totalFitness = parent1.fitness + parent2.fitness;
        if (totalFitness == 0) totalFitness = 1;
        double p1Weight = parent1.fitness / totalFitness;
        for (int i = 0; i < 5; i++) {
            offspring.genes[i] = rand.nextDouble() < p1Weight ? parent1.genes[i] : parent2.genes[i];
        }
        offspring.fitness = offspring.calculateFitness();
        return offspring;
    }

    // мутация - Каждый бит некоторых наименее пригодных потомков мутирует с некоторой вероятностью p

    private static void mutate(Individual ind, Random rand) {
        for (int i = 0; i < 5; i++) {
            if (rand.nextDouble() < MUTATION_PROBABILITY) {
                ind.genes[i] = rand.nextInt(RANGE_MAX - RANGE_MIN + 1) + RANGE_MIN;
            }
        }
        ind.fitness = ind.calculateFitness();
    }

    // замещение - Случайно пропорциональным образом создать новую популяцию из предков и потомков

    private static Individual[] replacePopulation(Individual[] oldPop, Individual[] newPop, Random rand) {
        // объединяем старую и новую популяции
        Individual[] combined = new Individual[POPULATION_SIZE * 2];
        System.arraycopy(oldPop, 0, combined, 0, POPULATION_SIZE);
        System.arraycopy(newPop, 0, combined, POPULATION_SIZE, POPULATION_SIZE);

        // сортируем по приспособленности (по убыванию)
        Arrays.sort(combined, (a, b) -> Double.compare(b.fitness, a.fitness));

        // выбираем пропорционально приспособленности
        Individual[] result = new Individual[POPULATION_SIZE];
        double totalFitness = Arrays.stream(combined).mapToDouble(ind -> ind.fitness).sum();
        if (totalFitness == 0) totalFitness = 1;

        for (int i = 0; i < POPULATION_SIZE; i++) {
            double r = rand.nextDouble() * totalFitness;
            double sum = 0;
            for (int j = 0; j < POPULATION_SIZE * 2; j++) {
                sum += combined[j].fitness;
                if (sum >= r) {
                    result[i] = new Individual(rand); // Создаем нового индивида
                    System.arraycopy(combined[j].genes, 0, result[i].genes, 0, 5);
                    result[i].fitness = combined[j].calculateFitness(); // Пересчитываем фитнес
                    break;
                }
            }
            // Гарантируем, что элемент не останется null
            if (result[i] == null) {
                result[i] = new Individual(rand);
            }
        }
        return result;
    }

    public static void main(String[] args) {
        Random rand = new Random();
        // инициализация начальной популяции
        Individual[] population = new Individual[POPULATION_SIZE];
        for (int i = 0; i < POPULATION_SIZE; i++) {
            population[i] = new Individual(rand);
        }

        double bestFitness = Double.NEGATIVE_INFINITY;
        Individual bestIndividual = null;
        int stagnationCount = 0;
        double previousBestFitness = Double.NEGATIVE_INFINITY;

        // основной цикл генетического алгоритма
        for (int generation = 0; generation < MAX_GENERATIONS; generation++) {
            // оценка приспособленности
            for (Individual ind : population) {
                if (ind != null) { // Добавлена проверка на null
                    ind.fitness = ind.calculateFitness();
                    if (ind.fitness > bestFitness) {
                        bestFitness = ind.fitness;
                        bestIndividual = new Individual(rand);
                        System.arraycopy(ind.genes, 0, bestIndividual.genes, 0, 5);
                        bestIndividual.fitness = ind.fitness;
                    }
                }
            }

            // Механизм вымирания
            if (Math.abs(bestFitness - previousBestFitness) < EXTINCTION_THRESHOLD) {
                stagnationCount++;
            } else {
                stagnationCount = 0;
            }
            previousBestFitness = bestFitness;

            if (stagnationCount >= STAGNATION_LIMIT) {
                // Замена части популяции новыми случайными особями
                int extinctionCount = (int) (POPULATION_SIZE * EXTINCTION_RATE);
                for (int i = 0; i < extinctionCount; i++) {
                    population[rand.nextInt(POPULATION_SIZE)] = new Individual(rand);
                }
                stagnationCount = 0;
            }

            // Селекция и скрещивание
            Individual[] newPopulation = new Individual[POPULATION_SIZE];
            for (int i = 0; i < POPULATION_SIZE; i++) {
                Individual parent1 = tournamentSelection(population, rand);
                Individual parent2 = tournamentSelection(population, rand);
                newPopulation[i] = crossover(parent1, parent2, rand);
            }

            // Мутация наименее приспособленных
            for (Individual ind : newPopulation) {
                if (ind.fitness < bestFitness * 0.5) {
                    mutate(ind, rand);
                }
            }

            // Замещение популяции
            population = replacePopulation(population, newPopulation, rand);

            // Вывод прогресса каждые 100 поколений
            if (generation % 100 == 0) {
                System.out.printf("Generation %d: Best Fitness = %.2f, Best Individual = %s%n",
                        generation, bestFitness, Arrays.toString(bestIndividual.genes));
            }
        }

        // Вывод финальных результатов
        System.out.printf("Final Best Fitness: %.2f%n", bestFitness);
        System.out.printf("Best Solution: x1=%d, x2=%d, x3=%d, x4=%d, x5=%d%n",
                bestIndividual.genes[0], bestIndividual.genes[1], bestIndividual.genes[2],
                bestIndividual.genes[3], bestIndividual.genes[4]);
        // Расчет значений уравнений для лучшего решения
        double[] finalResults = new double[3];
        finalResults[0] = Math.pow(bestIndividual.genes[1], 2) * bestIndividual.genes[2] * Math.pow(bestIndividual.genes[3], 2) * bestIndividual.genes[4] +
                Math.pow(bestIndividual.genes[1], 2) * Math.pow(bestIndividual.genes[2], 2) * Math.pow(bestIndividual.genes[3], 2) * bestIndividual.genes[4] +
                Math.pow(bestIndividual.genes[0], 2) * bestIndividual.genes[1] * bestIndividual.genes[2] +
                Math.pow(bestIndividual.genes[1], 2) * Math.pow(bestIndividual.genes[2], 2) * Math.pow(bestIndividual.genes[3], 2) * Math.pow(bestIndividual.genes[4], 2) +
                bestIndividual.genes[3] * bestIndividual.genes[4];
        finalResults[1] = bestIndividual.genes[0] * bestIndividual.genes[1] * Math.pow(bestIndividual.genes[2], 2) * Math.pow(bestIndividual.genes[3], 2) +
                bestIndividual.genes[1] * Math.pow(bestIndividual.genes[2], 2) * bestIndividual.genes[3] +
                bestIndividual.genes[3] +
                Math.pow(bestIndividual.genes[0], 2) * bestIndividual.genes[1] * Math.pow(bestIndividual.genes[3], 2) * Math.pow(bestIndividual.genes[4], 2);
        finalResults[2] = bestIndividual.genes[2] * bestIndividual.genes[3] * Math.pow(bestIndividual.genes[4], 2) +
                bestIndividual.genes[4] +
                Math.pow(bestIndividual.genes[0], 2) * Math.pow(bestIndividual.genes[1], 2) * Math.pow(bestIndividual.genes[2], 2) * bestIndividual.genes[3] +
                Math.pow(bestIndividual.genes[0], 2) * Math.pow(bestIndividual.genes[2], 2) * bestIndividual.genes[3] +
                Math.pow(bestIndividual.genes[2], 2) * bestIndividual.genes[3];
        System.out.printf("Equation 1 Result: %.2f (Target: -5)%n", finalResults[0]);
        System.out.printf("Equation 2 Result: %.2f (Target: 49)%n", finalResults[1]);
        System.out.printf("Equation 3 Result: %.2f (Target: -50)%n", finalResults[2]);
    }
}