# Multi Agent Dueling Deep Q-Network

The goal for this project is to implement a machine learning model that could play a game against another version of itself (learning separately so different strategies can exist) to learn until it becomes really good at the game. I also wanted this game to have random elements to it and have imperfect information for the agents.

### Random Elements in the game

This serves two porpouses:

1. **Help with learning**. Since a neural network is a deterministic algorithm (i.e. if you run 2 identical inputs through it, the output would be the same), if you make 2 neural networks play against each other, the game played will always be the same every time if there's no randomness in the environment itself.
2. **Making the envirorment harder**. I didn't want the game the agents played to be trivial. Adding randomness to it gives an "expected value" aspect to the strategy (i.e. the agents need to play knowing that the moves might not turn out exactly as they planned, changing the optimal strategy).

### Imperfect Information

A game with imperfect information means that each player has access to different information. One example of this is Poker, where each player can only know the cards on the table and on their hand, but not in the other player hand. I really like games with this characteristic, so I wanted a game with this aspect to it as well.

## The game

One of the easiest games to implement and play is Tic-tac-toe. The problem is that it is very easy to solve, doesn't have randomness, and it has perfect information. That won't work here.

I then created a new version of Tic-tac-toe that changes all of that.

### Modified Tic Tac Toe

<p align="center">
  <img src="https://i.imgur.com/6BaW8hA.png" width=600/>
</p>

The objective of the game still the same: have a straight line of 3 symbols in the board. The thing is that you can't simply draw your symbol into the square you want. For each draw action, there's a probability of that action succeeding that is defined by the "success probability of that square".

Each square has a different "success probability" and these probabilities are hidden from the players in the beginning of the game. These probabilities also change from game to game.

Along with the "drawing" action (where you try to draw your symbol into a square), you can also do a "checking" action, where you can check what's the "success probability" of a given square. This action always succeeds, and only the player who used the action has access to this information.

<p align="center">
    <img src="https://i.imgur.com/fWOCmJS.png" width=500>
</p>

## Training

In each game of training, firstly a random agent between `[DDDQN Agent 1], [DDDQN Agent 2], [Random Agent]` was chosen to play as `X` and then another random agent was chosen to play as `O`. They then played the game and the DDDQN agents learned from that.

After every 250 training games, 1000 games were played between each pair os agents (`[DDDQN Agent 1] vs [DDDQN Agent 2]`, `[DDDQN Agent 1] vs [Random Agent]` and `[DDDQN Agent 2] vs [Random Agent]`) and the win rate for each player, draw percentage and invalid games percentage (games ended with one of the players playing an invalid move) was recorded. These are shown below:

<table>
  <tbody>
    <tr>
      <td width="50%">
        <p align="center">
            DDDQN Agent 1 vs Random Agent
        </p>
        <p align="center">
            <img src="https://i.imgur.com/f1jTOzo.png" width="500"/>
        </p>
        <p align="center">
          <em>In gray, we can see that the invalid games quickly decreased to values close the 0, meaning that the agent learned how to not play invalid moves. It's winrate (in green) also reached 90% when playing against the random agent.</em>
        </p>
      </td>
      <td width="50%">
        <p align="center">
            DDDQN Agent 2 vs Random Agent
        </p>
        <p align="center">
            <img src="https://i.imgur.com/GGJRSjN.png" width="500"/>
        </p>
        <p align="center">
          <em>The training for the Agent 2 was pretty much the same as the Agent 1.</em>
          <br/><br/><br/>
        </p>
      </td>
    </tr>
    <tr>
       <td>
         <p align="center">
            DDDQN Agent 1 vs DDDQN Agent 2
        </p>
        <p align="center">
            <img src="https://i.imgur.com/VSABa0j.png" width="500"/>
        </p>
        <p align="center">
            <em>Now for the AI battle. The agents traded blows after they learned how to not invalidate the game. Sometimes one of them was better but the other quickly learns how to counter it. The draw percentage also grows as they find more and more efficient strategies.</em>
        </p>
      </td>
    </tr>
  </tbody>
</table>

## Next Steps

* Think of a cooler envirorment to make the agents play in.
* Play around with another learning algorithm like PPO

## References

Some resources I used while building this!

* [Tensorflow DQN Tutorial](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial)
* [PettingZoo project](https://pettingzoo.farama.org)
* Youtube Channel [Machine Learning with Phil](https://www.youtube.com/@MachineLearningwithPhil)
* [Gymnasium project](https://gymnasium.farama.org)