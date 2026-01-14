# RoboKino

## Abstract

Scalable dual-arm control represents a critical bottleneck in general-purpose robotics. While realizing emergent capabilities is essential for mastering complex dual-arm manipulation, deployment across diverse scenes is still constrained by limited adaptability and fragmented evaluation frameworks. To address these challenges, we present RoboKino, a novel simulation benchmark that introduces a unified paradigm for rigorously assessing dual-arm policies, leveraging domain-randomized task diversification and high-fidelity data synthesis. RoboKino designs five dual-arm atomic controllers that can be composed into collaborative complex tasks, alongside a standardized dataset of over 1,000 trajectories compliant with LeRobot v2.1 and augmented by 500 reusable interactive assets. Its integrated data acquisition, training, and inference pipeline enables robust policy generalization under realistic sensory inputs, including proprioceptive and visual observations that mirror Aloha-style hardware. Extensive empirical validation demonstrates that RoboKino significantly outperforms state-of-the-art methods in task completion efficacy and adaptability, establishing a solid foundation for scalable robotic systems and ensuring community-wide reproducibility.

## License

MIT License

Copyright (c) 2026 vigorlee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
