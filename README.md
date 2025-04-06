# SocialGPT

Start with emotion tokenization in interactions and test with a simple negotiation scenario.

- Start with basic emotion model like the Circumplex Model.
- Later, test more complex models such as Dimensional Models, Plutchik's Wheel, and Lazarus' Component Process.

Integrate a sentiment analysis model into the feedback system to guide emotional responses.

Balance logic and emotion by adding an emotional regulation component and measuring emotional intelligence (EQ).

Simulate and iterate, refining agent behavior as they engage in increasingly complex negotiations.

## Emotion Models

Many approaches have been proposed for modelling emotions:

| **Model**                 | **Key Idea**                              | **Advantages**                         | **Disadvantages**                    |
|---------------------------|-------------------------------------------|----------------------------------------|--------------------------------------|
| **James-Lange**            | Emotions result from physical reactions.  | Simple, biological basis               | Doesn't explain complex emotions.    |
| **Cannon-Bard**            | Emotions and bodily responses happen simultaneously. | Quick emotional response               | Ignores cognitive processes.         |
| **Schachter-Singer**       | Emotions arise from arousal + cognitive appraisal. | Explains mixed emotions                | Overemphasizes cognitive interpretation. |
| **Circumplex Model**       | Emotions on a 2D plane: valence (pleasant vs. unpleasant) and arousal. | Scalable, good for AI modeling         | Oversimplifies complex emotions.     |
| **Plutchik’s Wheel**       | 8 primary emotions in pairs, can combine to form complex emotions. | Detailed, accounts for emotional intensity | Complex to model, may miss nuance.  |
| **Ekman’s Basic Emotions** | 6 basic universal emotions (happiness, sadness, etc.). | Universally recognized, clear categories | Oversimplifies, misses cultural variation. |
| **Lazarus’ Component Process** | Emotions from cognitive appraisal based on goals and context. | Context-aware, individual differences  | Doesn't account for automatic emotions. |
| **Dimensional Models**     | Emotions mapped on 3 axes: valence, arousal, dominance. | Richer framework, handles power dynamics | Still overly dimensional, may miss nuance. |
| **FACS**                   | Emotions linked to facial expressions.    | Precise, useful in emotion recognition | Limited to facial expressions, misses internal states. |

## Emotion Tokenization

Emotion tokens can be included as part of the general embeddings table for the model.

When using a Circumplex Model for emotions, two major alternatives exist:

1. Splitting valence and arousal into two separate sets of tokens, one-hot encoded based on intensity. E.g. if there are 10 levels of valence and 10 levels of arousal, add a total of 20 emotion tokens.
2. Combining valence and arousal into a single set of tokens, one-hot encoded for each possible combination of levels of valence and arousal. E.g. add a total of `10 * 10 = 100` tokens.
