################################################################################
# 6.100B Spring 2023
# Problem Set 1
# Name: Justin Chen
# Collaborators: None
# Time: 6 hours

from state import State
import os

##########################################################################################################
## Problem 1
##########################################################################################################

def load_election_data(filename):
    """
    Reads the contents of a file, with data given in the following tab-separated format:
    State[tab]Democrat_votes[tab]Republican_votes[tab]EC_votes

    Please ignore the first line of the file, which are the column headers, and remember that
    the special character for tab is '\t'

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a list of State instances
    """

    # TODO
    
    f = open(filename, 'r')
    f_lines = f.readlines()
    
    f_lines.pop(0)
    f_split_lines = [line.strip('\n').split('\t') for line in f_lines]
    ret_list = []
    for line in f_split_lines:
        
        new_state = State(line[0], line[1], line[2], line[3])
        ret_list.append(new_state)
    f.close()
    return ret_list
    


##########################################################################################################
## Problem 2: Helper functions
##########################################################################################################

def get_election_winner(election_states):
    """
    Finds the winner of the election based on who has the most amount of EC votes.
    Note: In this simplified representation, all of EC votes from a state go
    to the party with the majority vote.

    Parameters:
    election_states - a list of State instances

    Returns:
    a tuple, (winner, loser) of the election i.e. ('dem', 'rep') if Democrats won, else ('rep', 'dem')
    """

    # TODO
    dem_sum = 0
    rep_sum = 0
    for state in election_states:
        winning_party = state.get_winner()
        ec_votes = state.get_ecvotes()
        if winning_party == "dem":
            dem_sum += ec_votes
        else:
            rep_sum += ec_votes
    if dem_sum > rep_sum:
        return ('dem', 'rep')
    else:
        return ('rep', 'dem')


def get_winning_states(election_states):
    """
    Finds the list of States that were won by the winning candidate (lost by the losing candidate).

    Parameters:
    election_states - a list of State instances

    Returns:
    A list of State instances won by the winning candidate
    """

    # TODO
    winner = get_election_winner(election_states)[0]
    winning_states = []
    for state in election_states:
        if state.get_winner() == winner:
            winning_states.append(state)
    return winning_states



def ec_votes_needed_to_flip(election_states, total=538):
    """
    Finds the number of additional EC votes required by the loser to change election outcome.
    Note: A party wins when they earn half the total number of EC votes plus 1.

    Parameters:
    election_states - a list of State instances
    total - total possible number of EC votes

    Returns:
    int, number of additional EC votes required by the loser to change the election outcome
    """

    # TODO
    margin = 0
    for state in election_states:
        winning_party = state.get_winner()
        ec_votes = state.get_ecvotes()
        if winning_party == "dem":
            margin += ec_votes
        else:
            margin -= ec_votes
    return abs(margin)//2 + 1


##########################################################################################################
## Problem 3: Brute Force approach
##########################################################################################################

def combos(L):
    """
    Helper function to generate powerset of all possible combinations
    of items in input list L. E.g., if
    L is [1, 2] it will return a list with elements
    [], [1], [2], and [1,2].

    DO NOT MODIFY THIS.

    Parameters:
    L - list of items

    Returns:
    a list of lists that contains all possible
    combinations of the elements of L
    """

    def get_binary_rep(n, num_digits):
        """
        Inner function to get a binary representation of items to add to a subset,
        which combos() uses to construct and append another item to the powerset.

        DO NOT MODIFY THIS.

        Parameters:
        n and num_digits are non-negative ints

        Returns:
            a num_digits str that is a binary representation of n
        """
        result = ''
        while n > 0:
            result = str(n%2) + result
            n = n//2
        if len(result) > num_digits:
            raise ValueError('not enough digits')
        for i in range(num_digits - len(result)):
            result = '0' + result
        return result

    powerset = []
    for i in range(0, 2**len(L)):
        binStr = get_binary_rep(i, len(L))
        subset = []
        for j in range(len(L)):
            if binStr[j] == '1':
                subset.append(L[j])
        powerset.append(subset)
    return powerset


def brute_force_swing_states(winner_states, ec_votes_needed):
    """
    Finds a subset of winner_states that would change an election outcome if
    voters moved into those states, these are our swing states. Iterate over
    all possible move combinations using the helper function combos(L).
    Return the move combination that minimises the number of voters moved. If
    there exists more than one combination that minimises this, return any one of them.

    Parameters:
    winner_states - a list of State instances that were won by the winning candidate
    ec_votes_needed - int, number of EC votes needed to change the election outcome

    Returns:
    * A tuple containing the list of State instances such that the election outcome would change if additional
      voters relocated to those states, as well as the number of voters required for that relocation.
    * A tuple containing the empty list followed by zero, if no possible swing states.
    """

    # TODO
    subsets = combos(winner_states)
    good_combos = []
    for combo in subsets:
        ec_sum = 0
        for state in combo:
            ec_sum += state.get_ecvotes()
        if ec_sum >= ec_votes_needed:
            good_combos.append(combo)
    min_margin = -1
    min_combo = []
    for combo in good_combos:
        margin_total = 0
        for state in combo:
            margin_total += state.get_margin()+1
        if (min_margin == -1) or (margin_total < min_margin):
            min_margin = margin_total
            min_combo = combo
    
    return (min_combo, min_margin)



##########################################################################################################
## Problem 4: Dynamic Programming
## In this section we will define two functions, max_voters_moved and min_voters_moved, that
## together will provide a dynamic programming approach to find swing states. This problem
## is analagous to the complementary knapsack problem, you might find Lecture 1 of 6.100B useful
## for this section of the pset.
##########################################################################################################


def max_voters_moved(winner_states, max_ec_votes, memo=None):
    """
    Finds the largest number of voters needed to relocate to get at most max_ec_votes
    for the election loser.

    Analogy to the knapsack problem:
        Given a list of states each with a weight(ec_votes) and value(margin+1),
        determine the states to include in a collection so the total weight(ec_votes)
        is less than or equal to the given limit(max_ec_votes) and the total value(voters displaced)
        is as large as possible.

    Parameters:
    winner_states - a list of State instances that were won by the winner
    max_ec_votes - int, the maximum number of EC votes

    Returns:
    * A tuple containing the list of State instances such that the maximum number of voters need to
      be relocated to these states in order to get at most max_ec_votes, and the number of voters
      required required for such a relocation.
    * A tuple containing the empty list followed by zero, if every state has a # EC votes greater
      than max_ec_votes.
    """

    # TODO
    #important base case
    if winner_states == []:
        return [], 0
    # initiation of memo
    if memo == None:
        memo = {}
    # check for memoization
    if (len(winner_states), max_ec_votes) in memo:
        return memo[(len(winner_states), max_ec_votes)]
    
    current_state = winner_states[0]
    if current_state.get_ecvotes() > max_ec_votes:
        solution, value = max_voters_moved(winner_states[1:], max_ec_votes, memo)
    else:
        # left branch with the state
        sol_with, val_with = max_voters_moved(winner_states[1:], max_ec_votes-current_state.get_ecvotes(), memo)
        sol_with = sol_with + [current_state]
        val_with = val_with + current_state.get_margin() + 1
        # right branchw without state
        sol_wo, val_wo = max_voters_moved(winner_states[1:], max_ec_votes, memo)
        #find better branch
        solution, value = (sol_with, val_with) if val_with > val_wo else (sol_wo, val_wo)
    # update memo with best combo
    memo[(len(winner_states), max_ec_votes)] = solution, value
    return solution,value



def min_voters_moved(winner_states, ec_votes_needed):
    """
    Finds a subset of winner_states that would change an election outcome if
    voters moved into those states. Should minimize the number of voters being relocated.
    Only return states that were originally won by the winner (lost by the loser)
    of the election.

    Hint: This problem is simply the complement of max_voters_moved. You should call
    max_voters_moved with ec_vote_limit set to (#ec votes won by original winner - ec_votes_needed)

    Parameters:
    winner_states - a list of State instances that were won by the winner
    ec_votes_needed - int, number of EC votes needed to change the election outcome

    Returns:
    * A tuple containing the list of State instances (which we can call swing states) such that the
      minimum number of voters need to be relocated to these states in order to get at least
      ec_votes_needed, and the number of voters required for such a relocation.
    * * A tuple containing the empty list followed by zero, if no possible swing states.
    """

    # TODO
    won_ec_votes = sum(state.get_ecvotes() for state in winner_states)
    min_states = [state for state in winner_states if state not in max_voters_moved(winner_states, won_ec_votes-ec_votes_needed)[0]]
    if min_states == []:
        return [], 0
    else:
        voters_moved = sum(state.get_margin()+1 for state in min_states)
        return min_states, voters_moved


##########################################################################################################
## Problem 5
##########################################################################################################


def relocate_voters(election_states, swing_states, ideal_states = ['AL', 'AZ', 'CA', 'TX']):
    """
    Finds a way to shuffle voters in order to flip an election outcome. Moves voters
    from states that were won by the losing candidate (states not in winner_states), to
    each of the states in swing_states. To win a swing state, you must move (margin + 1)
    new voters into that state. Any state that voters are moved from should still be won
    by the loser even after voters are moved. Also finds the number of EC votes gained by
    this rearrangement, as well as the minimum number of voters that need to be moved.
    Note: You cannot move voters out of Alabama, Arizona, California, or Texas.

    Parameters:
    election_states - a list of State instances representing the election
    swing_states - a list of State instances where people need to move to flip the election outcome
                   (result of min_voters_moved or brute_force_swing_states)
    ideal_states - a list of Strings holding the names of states where residents cannot be moved from
                   (default states are AL, AZ, CA, TX)

    Return:
    * A tuple that has 3 elements in the following order:
        - an int, the total number of voters moved
        - an int, the total number of EC votes gained by moving the voters
        - a dictionary with the following (key, value) mapping:
            - Key: a 2 element tuple of str, (from_state, to_state), the 2 letter State names
            - Value: int, number of people that are being moved
    * None, if it is not possible to sway the election
    """
    # TODO
    # getting losing candidate states that are not ideal
    winning_states = get_winning_states(election_states)
    losing_candidate_states = [state for state in election_states if (state not in winning_states) and (state.get_name() not in ideal_states)]
    # check if not possible tos way
    if sum(state.get_margin()-1 for state in losing_candidate_states) < sum(state.get_margin()+1 for state in swing_states):
        return None
    #creating return dict and a copy of swing states
    mapping_dict = {}
    swing_states_list = {state.get_name(): state.get_margin()+1 for state in swing_states}
    loser_states_list = {state.get_name(): state.get_margin()-1 for state in losing_candidate_states}
    while len(swing_states_list) > 0:
        if len(loser_states_list) == 0:
            return None 
        if list(swing_states_list.values())[0] > list(loser_states_list.values())[0]:
            mapping_dict[(list(loser_states_list.keys())[0], list(swing_states_list.keys())[0])] = list(loser_states_list.values())[0]
            swing_states_list[list(swing_states_list.keys())[0]] -= list(loser_states_list.values())[0]
            loser_states_list.pop(list(loser_states_list.keys())[0])
        else:
            mapping_dict[(list(loser_states_list.keys())[0], list(swing_states_list.keys())[0])] = list(swing_states_list.values())[0]
            loser_states_list[list(loser_states_list.keys())[0]] -= list(swing_states_list.values())[0]
            swing_states_list.pop(list(swing_states_list.keys())[0])
    
    return (sum(state.get_margin()+1 for state in swing_states), sum(state.get_ecvotes() for state in swing_states), mapping_dict)


if __name__ == "__main__":
    
    # Uncomment the following lines to test each of the problems

    # tests Problem 1
    year = 2012
    election_states = load_election_data(f"{year}_results.txt")
    print(len(election_states))
    print(election_states[0])

    # tests Problem 2
    winner, loser = get_election_winner(election_states)
    won_states = get_winning_states(election_states)
    names_won_states = [state.get_name() for state in won_states]
    reqd_ec_votes = ec_votes_needed_to_flip(election_states)
    print("Winner:", winner, "\nLoser:", loser)
    print("States won by the winner: ", names_won_states)
    print("EC votes needed:",reqd_ec_votes, "\n")

    # tests Problem 3
    brute_election = load_election_data("6100B_results.txt")
    brute_won_states = get_winning_states(brute_election)
    brute_ec_votes_needed_to_flip = ec_votes_needed_to_flip(brute_election, total=14)
    brute_swing, voters_brute = brute_force_swing_states(brute_won_states, brute_ec_votes_needed_to_flip)
    names_brute_swing = [state.get_name() for state in brute_swing]
    ecvotes_brute = sum([state.get_ecvotes() for state in brute_swing])
    print("Brute force swing states results:", names_brute_swing)
    print("Brute force voters displaced:", voters_brute, "for a total of", ecvotes_brute, "Electoral College votes.\n")

    # tests Problem 4a: max_voters_moved
    print("max_voters_moved")
    total_lost = sum(state.get_ecvotes() for state in won_states)
    non_swing_states, max_voters_displaced = max_voters_moved(won_states, total_lost-reqd_ec_votes)
    non_swing_states_names = [state.get_name() for state in non_swing_states]
    max_ec_votes = sum([state.get_ecvotes() for state in non_swing_states])
    print("States with the largest margins (non-swing states):", non_swing_states_names)
    print("Max voters displaced:", max_voters_displaced, "for a total of", max_ec_votes, "Electoral College votes.", "\n")

    # tests Problem 4b: min_voters_moved
    # for 4c need just one less ec_vote, so subtract 1 from reqd_ec_votes two lines down
    print("min_voters_moved")
    swing_states, min_voters_displaced = min_voters_moved(won_states, reqd_ec_votes)
    swing_state_names = [state.get_name() for state in swing_states]
    swing_ec_votes = sum([state.get_ecvotes() for state in swing_states])
    print("Complementary knapsack swing states results:", swing_state_names)
    print("Min voters displaced:", min_voters_displaced, "for a total of", swing_ec_votes, "Electoral College votes. \n")

    # tests Problem 5: relocate_voters
    print("relocate_voters")
    flipped_election = relocate_voters(election_states, swing_states)
    print("Flip election mapping:", flipped_election)