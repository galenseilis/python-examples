import random
import csv


def simulate_monty_hall(num_simulations, csv_filename="monty_hall_simulation.csv"):
    results = []

    for _ in range(num_simulations):
        # Step 1: Randomly place the car behind one of the 3 doors (0, 1, or 2)
        car_position = random.randint(0, 2)

        # Step 2: The player randomly chooses one of the 3 doors (initial guess)
        player_choice = random.randint(0, 2)

        # Step 3: Determine if the initial guess was correct
        init_win = 1 if player_choice == car_position else 0

        # Step 4: Monty opens one of the other doors that does not have the car behind it
        available_doors = [
            door for door in range(3) if door != player_choice and door != car_position
        ]
        monty_opens = random.choice(available_doors)

        # Step 5: Player decides whether to switch doors or not
        # Randomly decide whether to switch or stay (1 = switch, 0 = stay)
        switch = random.choice([0, 1])

        if switch:
            # Player switches to the remaining unopened door
            player_choice = next(
                door
                for door in range(3)
                if door != player_choice and door != monty_opens
            )

        # Step 6: Determine if the final choice was correct
        final_win = 1 if player_choice == car_position else 0

        # Save the result of this simulation
        results.append([init_win, switch, final_win])

    # Write the results to a CSV file
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["init_win", "decision", "final_win"])
        # Write the simulation results
        writer.writerows(results)

    print(f"Simulation completed and saved to {csv_filename}")


# Run the simulation for 10000 trials
simulate_monty_hall(10000)
