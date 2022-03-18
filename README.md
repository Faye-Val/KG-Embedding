### DeepWalk

Update the implementation of deepwalk.  


##### References:
>The deep walk is an algorithm proposed for learning latent representations of vertices in a network.  

DeepWalk Algorithm:

- random walk generator takes a graph G as inputs and return random samples of the path in G with the given path length.
- Use Skip-gram like methods to learn representations of vertices with the above paths.