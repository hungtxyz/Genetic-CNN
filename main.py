from genetic_algthm.population import Population

if __name__ == "__main__":
    pop = Population(num_individual=8, num_phase=3, num_conv=19, num_epochs=7)
    print(pop.individuals)

    num_generation = 10
    for g in range(0, num_generation):
        pop.set_epoch(g)
        pop.save_checkpoint("./pop_checkpoint", "./fit_checkpoint")
        pop.write_log()
        print("gen ", g, pop.fitness)

        pop.crossover(0.5)
        pop.mutate(0.001)
        pop.select()
