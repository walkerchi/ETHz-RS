
# ETHz Robotics Seminar

---

keywords: sensor, reinforcement learning, sim2real

---

supervisor: Arm  Philip, Bjelonic Filip

---

## Concept:

- **simulation bias** :  The resulting policies may work well with the forward model (i.e. the simulator) but poorly on the real system

- **mental rehearsal** : In robot reinforcement learning, the learning step on the simulated system is often called “mental rehearsal”

1. [Reinforcement learning in robotics: A survey](https://journals.sagepub.com/doi/full/10.1177/0278364913495721?casa_token=ZHHJVDn7ds8AAAAA%3AH6Vg8LSPBliwrLS1xiMmUO-qslz1ZQ76U7sxUQpDcdqms7z4tASOHcrM3j_VLg4wuUClaBH8WnsI)
In the **survey** paper, the Section 3.2-3.4 illustrate on the environment sampling, modeling uncertainty and goal specification for reinforcement learning on robotics, which is related to the sensor sim2real.

2. [Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303)
The **survey** paper provide a collection of sim2real works. It include some works like Domain Randomization, which could be a good start point for different method used in sim2real.

3. [Parallel Learning: Overview and Perspective for Computational Learning Across Syn2Real and Sim2Real](https://ieeexplore.ieee.org/document/10057176)
This **survey** paper provides a detailed overview of the virtual-to-real paradigm, which involves Parallel Intelligence, Digital Twin, Syn-to-Real and Sim-to-Real. It's also a good start point.

4. [Towards Closing the Sim-to-Real Gap in Collaborative Multi-Robot Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9310796?casa_token=qGDV_tCatxMAAAAA:5zBHDB7jau8xnP5qx41yhNKzehK1gjID4hLRhxDWLTw8EXajnt4ZpV48DTVZqVsk4J2hT9nlVg)
This research investigates the sim2real gap in **multi-robot systems** using multi-agent reinforcement learning, specifically focusing on the effects of sensing and calibration discrepancies on collective learning with PPO.

5. [Sim-to-Real: Learning Agile Locomotion For Quadruped Robots](https://arxiv.org/abs/1804.10332)
Introduce an automated strategy for learning agile locomotion in quadruped robots. The approach enhances simulation fidelity with **system identification**, **precise actuator modeling**, and **latency simulation**. Robust control is achieved by varying the simulation environment, introducing **disturbances**, and utilizing a streamlined observation space.


6. [Dynamics Randomization Revisited: A Case Study for Quadrupedal Locomotion](https://ieeexplore.ieee.org/abstract/document/9560837?casa_token=yphRgw3TAuMAAAAA:gIwqpF0vq51bQ-1Tzm3C6ZyWFd4KlmzmHjAJE4o9fGy6KN9TLhx_PCNiUg5lp21DU12tzZVECA)
The authors find that effective sim-to-real policy transfer can occur **without dynamics randomization** or on-robot adaptation. Through comprehensive sim-to-sim and sim-to-real experiments, they investigate essential factors influencing successful policy transfer, including gait, speed, and stepping frequency.

7. [Reinforcement Learning with Perturbed Rewards ](https://ojs.aaai.org/index.php/AAAI/article/view/6086)
This paper presents a robust framework for addressing the issue of **noisy reward** signals in robotics, a common problem when transferring reinforcement learning models from simulation to real-world applications (sim2real). The authors develop a method to compensate for perturbed rewards without assuming noise distribution, demonstrating enhanced real-world performance of robotic systems with improved convergence in noisy environments.

8. [Sim-To-Real via Sim-To-Sim: Data-Efficient Robotic Grasping via Randomized-To-Canonical Adaptation Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/James_Sim-To-Real_via_Sim-To-Sim_Data-Efficient_Robotic_Grasping_via_Randomized-To-Canonical_Adaptation_Networks_CVPR_2019_paper.html)
The paper introduces **Randomized-to-Canonical Adaptation Networks** (RCANs), a new technique for bridging the gap between simulation and real-world data in robotics, without using any real-world data. RCANs transform simulated images into a standardized form, making it possible to train models in simulation and apply them in the real world, which is demonstrated by significantly improving the success rate of a vision-based robotic grasping task.

9. [Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience](https://ieeexplore.ieee.org/abstract/document/8793789?casa_token=BYg3gfZC2ZgAAAAA:cY2h-nusTELbjyc8YsOstIolZ-5PhXDF4Bg3ryvNhxID68D7UC4x0rrR1gLnWo1f3_Lp6lSR_g)
The study addresses transferring simulated training policies to real-world robotics by dynamically adjusting the simulation parameters based on a small number of **real-world trials**. This adaptive method aligns simulated policy behavior more closely with actual performance, enhancing the real-world transferability of the trained policies.

10. [Analysis of Randomization Effects on Sim2Real Transfer in Reinforcement Learning for Robotic Manipulation Tasks](https://ieeexplore.ieee.org/abstract/document/9981951?casa_token=H53nvLiwVpQAAAAA:kZ3LOD1bqUpaFSwIjF7mZwB8Q8mBnWlOeP9USL2Xajx_28imw6iYv4V-zz6MJM4FviNTk9R1yg)
To enable systematic evaluation of Sim2Real, the authors propose a reproducible experimental framework for a robotic reach-and-balance task, through which they assess four randomization strategies across three parameters. Their findings reveal that while increased randomization can facilitate Sim2Real transfer, it may also impede policy optimization in simulation; they also find that **full randomization** and **fine-tuning** yield the best real-world performance.

11. [Sim2Real Transfer for Deep Reinforcement Learning with Stochastic State Transition Delays](https://proceedings.mlr.press/v155/sandha21a.html)
The paper discusses the challenge of Sim2Real transfer in Deep Reinforcement Learning (RL) for robotics, particularly due to variations in sensor sampling rates and actuation delays. It points out the limitations of domain randomization in addressing the inconsistencies in state transition delays during real-world deployment. Introducing the Time-in-State RL (TSRL) approach, which incorporates **timing delays** and **sampling** rates into the training observations, the study shows improved robustness of Deep RL policies.

12. [DiAReL: Reinforcement Learning with Disturbance Awareness for Robust Sim2Real Policy Transfer in Robot Control](https://arxiv.org/abs/2306.09010)
The study presents a method for addressing delays in robotic control by enhancing state information with a history of recent actions, which satisfies the Markov property for Delayed Markov Decision Processes (DMDPs). To mitigate the challenges of direct training on robots, such as inefficiency and safety risks, the authors propose simulating robotic dynamics with consideration for **intrinsic uncertainties** as disturbances in system inputs. A novel representation called disturbance-augmented DMDP is introduced for training on-policy reinforcement learning algorithms, accounting for these disturbances.

13. [Reinforcement Learning with Adaptive Curriculum Dynamics Randomization for Fault-Tolerant Robot Control](https://arxiv.org/abs/2111.10005)
The research focuses on improving the fault tolerance of quadruped robots to actuator failure using an Adaptive Curriculum Reinforcement Learning algorithm with **Dynamics Randomization** (ACDR). This method trains robots to handle random actuator failures, developing a robust control policy without needing separate failure detection or policy switching mechanisms. 

14. [Automatic Gait Optimization with Gaussian Process Regression](https://webdocs.cs.ualberta.ca/~dale/papers/ijcai07a.pdf)
They introduce a Bayesian **optimization** method using Gaussian process regression, leveraging a global search strategy that incorporates all noisy gait evaluations. 

15. [Exploiting Model Uncertainty Estimates for Safe Dynamic Control Learning](https://proceedings.neurips.cc/paper_files/paper/1996/hash/93fb9d4b16aa750c7475b6d601c35c2c-Abstract.html)
A novel algorithm, drawing from dual control principles and Bayesian locally weighted regression models, is proposed to **balance the need for exploration** with the requirement to avoid overly risky actions. 

16. [How to Sim2Real with Gaussian Processes: Prior Mean versus Kernels as Priors](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1565302&dswid=-2444)
It suggests a shift from traditional methods of incorporating **prior knowledge** through a prior mean function or 'fake' data to a more flexible approach that embeds this knowledge directly into GP kernels.

17. [Learning Vision-Guided Quadrupedal Locomotion End-to-End with Cross-Modal Transformers](https://arxiv.org/abs/2107.03996)
The paper introduces LocoTransformer, an **end-to-end reinforcement learning method** that enhances quadrupedal robot locomotion by integrating proprioceptive data with visual inputs from depth sensors, using a Transformer-based model. This approach allows the robot to anticipate and navigate complex terrain proactively, rather than merely reacting to immediate contact.

18. [Crossing the Gap: A Deep Dive into Zero-Shot Sim-to-Real Transfer for Dynamics](https://ieeexplore.ieee.org/abstract/document/9341617?casa_token=kahI3DOTdtgAAAAA:ACANVD-saqdfmasVpnjfzZv2TZ7hzSc0t8DQ96ruMOj6JR3k9faeHs3uf_Q7q7wYs0Qd4hhveg)
It was found that a simple approach of introducing **random forces** in simulations was as effective as more complex strategies like randomizing dynamics parameters or adapting policies through recurrent networks.

19. [Trustworthy Reinforcement Learning Against Intrinsic Vulnerabilities:Robustness, Safety, and Generalizability](https://arxiv.org/abs/2209.08025)
This paper presents a comprehensive survey of **trustworthy reinforcement learning (RL)**, focusing on three core aspects: robust training, safety training, and generalizability. Thoese are three important aspects of sim2real which is also mentioned in the first survey.

20. [Discovering Blind Spots in Reinforcement Learning](https://arxiv.org/abs/1805.08966)
This paper proposes a method to identify and predict 'blind spots' in reinforcement learning (RL) agents that occur due to **insufficient state representations**, which can lead to costly mistakes when agents trained in simulation are deployed in the real world. To tackle this, the authors have developed a predictive model that uses oracle feedback—such as demonstrations and corrections—to learn about these blind spots.

21. [TuneNet: One-Shot Residual Tuning for System Identification and Sim-to-Real Robot Task Transfer](http://proceedings.mlr.press/v100/allevato20a.html)
TuneNet, a machine-learning-based method is designed to bridge the gap between simulated and real-world environments for robot training purposes. Unlike traditional approaches that require a lot of real-world data or simulation samples, TuneNet uses an iterative residual tuning technique that can adjust the parameters of a simulation model to more closely match the real world with just **a single observation** from the target environment and minimal simulation effort.

22. [Sim2Real2Sim: Bridging the Gap Between Simulation and Real-World in Flexible Object Manipulation](https://ieeexplore.ieee.org/abstract/document/9287921?casa_token=qKgRLYO7UP4AAAAA:YlgsFqqgCWHFbMZ9idWOA8zWcylL5jz-gZU_uSJ52OVkh1FoJ3-Sabbqj37i88a1dttBUBuIHQ)
The Sim2Real2Sim strategy outlined in the paper provides an innovative approach to improving the accuracy and utility of robotic manipulation tasks by continuously **updating and refining simulation models based on real-world performance**.

23. [Sim2Real Transfer for Reinforcement Learning without Dynamics Randomization](https://ieeexplore.ieee.org/abstract/document/9341260?casa_token=jGFad_P1MHUAAAAA:UYMglcjn4Wzra6j-oI3HlbBk6YUUmu7PUzwzHcOuebdkYImGq9LNftVFRB5BMm7YOFGSnYy48g)
The paper presents a method that combines Operational Space Control (OSC) with reinforcement learning to perform tasks in Cartesian space on a robotic system. This approach is OSC-driven, adaptable learning ensures precise, safe robot control with **seamless sim-to-real policy transfer** and constraints management.

24. [Preparing for the Unknown: Learning a Universal Policy with Online System Identification](https://arxiv.org/abs/1702.02453)
They train **robust control policies** for diverse dynamic models using a Universal Policy (UP) and Online System Identification (OSI), enabling adaptability to sudden environmental changes and narrowing the Reality Gap in simulations.

25. [Modelling Generalized Forces with Reinforcement Learning for Sim-to-Real Transfer](https://arxiv.org/abs/1910.09471)
The paper introduces a framework that augments the analytical model by optimizing **state-dependent generalized forces**, which can capture constraints in the motion equations while retaining clear physical meaning

26. [Learning Active Task-Oriented Exploration Policies for Bridging the Sim-to-Real Gap](https://arxiv.org/abs/2006.01952)
The paper proposes learning **exploration policies** that are focused on the task at hand. Instead of trying to be generally robust (as with domain randomization) or learning during task performance (as with online system-identification), this approach directs the exploration to gather information relevant to the task.

27. [Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion](https://proceedings.mlr.press/v205/fu23a.html)
The study introduces a reinforcement learning-based unified **control policy** for legged robots with arms, overcoming the limitations of traditional decoupled control systems that often result in uncoordinated and unnatural movements. By employing Regularized Online Adaptation and Advantage Mixing techniques, the policy bridges the Sim2Real gap and avoids training pitfalls like local minima, enabling dynamic, whole-body control.

28. [Accurate Dynamics Models for Agile Drone Flight:Zero-Shot Sim2Real-Transfer of Neural Controllers](https://transferabilityinrobotics.github.io/icra2023/spotlight/TRW02_abstract.pdf)
This paper presents a hybrid modeling approach for quadrotors that combines fundamental principles with data-driven methods, significantly enhancing model accuracy for robust control systems. This improved modeling enables the use of simulations to train neural controllers with reinforcement learning for **zero-shot real-world transfers** and to precisely verify controller performance.

29. [Unsupervised Domain Adaptation with Dynamics-Aware Rewards in Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f187a23c3ee681ef6913f31fd6d6446b-Abstract.html)
The paper proposes a method for unsupervised **domain adaptation** in reinforcement learning, which allows an agent to develop skills in a source environment and transfer them to a target environment with different dynamics. By using a KL-divergence regularized objective, the method rewards the agent for skill discovery and behavioral alignment across dynamic changes, leading to adaptive skill acquisition.

30. [Sim-to-real: Quadruped Robot Control with Deep Reinforcement Learning and Parallel Training](https://ieeexplore.ieee.org/abstract/document/10011921)
This paper introduces an **end-to-end neural network framework** that, contains an estimator network for estimating the robot's ontology states in addition to the critic and actor networks. These states serve as an important observation data for the critic and actor networks.

31. [Closing the Sim-to-Real Gap for Ultra-Low-Cost, Resource-Constrained, Quadruped Robot Platforms](http://jabbourjason.com/pubs/Closing_the_sim-to-real_gap_for_ultra-low-cost.pdf)
Their advancements facilitate the transfer of sophisticated locomotion skills to simpler robotic platforms, characterized by significant **hardware constraints** such as limited computational power, minimal sensory inputs, and basic actuation.

32.  [Generation of GelSight Tactile Images for Sim2Real Learning](https://ieeexplore.ieee.org/abstract/document/9369877?casa_token=uWLWHXhT6T0AAAAA:GMe81q1SjeeUOJZYf_8VP2MswEC0upSXZiyQAH4QuDHvRafLpb2mXiBE9hZtto7qPwAStWT1ZQ)
This simulated sensor facilitates Sim2Real learning by providing **tactile information** that complements vision, especially where visual occlusion by robot hands is an issue.

33. [i-Sim2Real: Reinforcement Learning of Robotic Policies in Tight Human-Robot Interaction Loops](https://proceedings.mlr.press/v205/abeyruwan23a.html)
This study tackles the challenge of sim-to-real transfer in the context of human-robot interaction (HRI), particularly focusing on the complex, dynamic task of robotic table tennis. The proposed Iterative-Sim-to-Real (i-S2R) method innovatively bridges the gap between simulation and real-world interaction by iteratively refining both the robot's policy and the simulated model of **human behavior** through cycles of simulation training and real-world deployment. 

34. [Self-improving Models for the Intelligent Digital Twin: Towards Closing the Reality-to-Simulation Gap](https://www.sciencedirect.com/science/article/pii/S2405896322001823)
This paper introduces a three-step reinforcement learning approach to refine **Digital Twin models** in Cyber-Physical Manufacturing Systems (CPMS). By using autonomous mobile robots as a test case, the method first aligns and synchronizes the real and simulated data, then employs reinforcement learning to detect discrepancies between the two, and finally adjusts the Digital Twin models to compensate for these differences. 

35. [Sim2Real Predictivity: Does Evaluation in Simulation Predict Real-World Performance?](https://ieeexplore.ieee.org/abstract/document/9158349?casa_token=rfkpcGKfLl4AAAAA:9wCizUDY4c567eO0o7POKb1FE8gBKJe6aR6UOFAEclMevrDvxo5Idz8fxO-PstzCWoyNEajqOg)
The paper indicating that simulator performance doesn't necessarily translate to the real world, often due to agents exploiting simulator flaws. It suggesting that careful **tuning of simulations** can make in-simulation testing more indicative of real-world performance.

36. [Learning Bipedal Walking for Humanoids With Current Feedback](https://ieeexplore.ieee.org/abstract/document/10201853)
The key idea is to utilize the current **feedback from the actuators** on the real robot, after training the policy in a simulation environment artificially degraded with poor torque-tracking.

37. [NeRF2Real: Sim2real Transfer of Vision-guided Bipedal Motion Skills using Neural Radiance Fields](https://ieeexplore.ieee.org/abstract/document/10161544?casa_token=oQw3teIEj7oAAAAA:20jGq2yt8H4WXwls6y4UM1NHfV29A8yZaCWaZb21MPBeb3pSQ-5VhDXn2T9SppNYCyr49L_yEQ)
The study introduces a technique that uses short videos captured by a smartphone to **create realistic sim2real environments** for training robotic policies using Neural Radiance Fields (NeRF) to learn scene geometry and enable novel view synthesis.

38. [Safety-Critical Controller Verification via Sim2Real Gap Quantification](https://ieeexplore.ieee.org/abstract/document/10161126?casa_token=yCF1f0R1Lq0AAAAA:xo21Cx53gsrg0mB7SfS8lFxmWvfgBTlWyQdvoWxFACMJjHAoMVwlxsPfTrfBeibqRw8u96Bu6A)
The authors have developed a method to detect and **measure the discrepancies** between simulation models and real-world system behavior, known as the sim2real gap. By incorporating this measured gap into their simulations, they create an "uncertain model" that more closely reflects reality. This model is then used to design and test controllers in the simulated environment with a probabilistic approach that ensures their effectiveness and reliability in the real world.


39. [Auto-Tuned Sim-to-Real Transfer](https://ieeexplore.ieee.org/abstract/document/9562091?casa_token=UVHKggFtRusAAAAA:TMmJo8MCWVG8EjiwmWRs9fR6moQ_3f9o5ngzyxzhOd0eGov085S_Tyk86BvDQODXAjZ6lKr25g)
The authors address the sim-to-real transfer problem by introducing a method that automatically adjusts simulation parameters to better match the real world, using only RGB images without requiring predefined rewards or state estimation. Their approach, based on a Search Param Model (SPM), treats parameter tuning as a search problem, **iteratively modifying simulation parameters to align with real-world values**.

40. [Adaptability Preserving Domain Decomposition for Stabilizing Sim2Real Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9341124?casa_token=FGwsOIdemZgAAAAA:Ua5r3OY7a9Jq918-Thrl-BHQ2435fAX8DJsPr3IdN7iBQo2nhVni9vd81za0nQhGCo8ylFUpuw)
The paper addresses the challenges in sim-to-real transfer for robot tasks, particularly the trade-off between adaptability and training stability when using **Domain Randomization** (DR). The authors propose a novel algorithm called Domain Decomposition (DD), which partitions the randomized environment into distinct segments and trains individual reinforcement learning policies for each one.

