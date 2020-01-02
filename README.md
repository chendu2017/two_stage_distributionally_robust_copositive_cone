# two_stage_distributionally_robust_copositive_cone
This is the code for one chapter of my master degree thesis. Basically, I want to design a network with two-stage location-inventory model, modifed by distribuitonally robust techniques. Copositive Cone (CP) programming is involved, and sime-definite matrix approximation is incorporated to deal the CP problem.

Basically, the idea comes from this paper [1]. And to run the code, the commercial software Mosek (www.mosek.com) is needed. If you are an academic member, the software company provides a free license for you.

In their work, they designed a network when the demand information in the network is very limited, i.e., only the first moment and the second moment are known to the decision maker. They proposed a distributionally robust model with 0-1 decision variables, since the only thing in their work is designing the network, or more specifically, the roads or the links. 

Enlighted by their work and another work ([2]), I want to design a efficient network by choosing locations to establish warehouses and storing materials there to match uncertain demand with prepositioned materials.

My problem is a two-stage problem. The first stage is a classical location-inventory problem. Namely, I need to select some places and reserve materials. And the second stage is an allocation problem. In other words, after the demand realizes, the decision maker needs to fulfill the demand as large as possible by stored materials. Because the coefficients of the second-stage objective function of objective is *1*, the right hand side of the dual problem are only 0 or 1. And by max-flow min-cut theorem, the dual problem has only 0 or 1 feasible solutions. Successfully, we change the model into a tractable model which has discussed in the paper [2]. Taking dual twice leads us to a tractable model (with 0-1 binary decision variable for the first stage decision). And for the final model, I coded it. This project is the code for the final model.

By the way, since the first stage involves binary decision variable (whether building a warehouse for a place), I introduce the branch and bound algorithm in my code. Some heuristic are proposed to accelerate the searching process. To my best knowledge, SCIP-SDP, a open-source fres software provides a powerful solver for semi-definite positive problems with binary variable. But for my thesis, I did not use this solver, instead, I build the integer solver from scratch by myself. 


[1]*Yan Z, Gao S Y, Teo C P. On the design of sparse but efficient structures in operations[J]. Management Science, 2017, 64(7): 3421-3445.*

[2]*Natarajan K, Teo C P, Zheng Z. Mixed 0-1 linear programs under objective uncertainty: A completely positive representation[J]. Operations research, 2011, 59(3): 713-728.*
