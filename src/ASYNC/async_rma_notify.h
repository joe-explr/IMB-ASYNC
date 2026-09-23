/*****************************************************************************
 *                                                                           *
 * Copyright 2016-2018 Intel Corporation.                                    *
 * Copyright 2019-2023 Alexey V. Medvedev                                    *
 *                                                                           *
 *****************************************************************************

   The 3-Clause BSD License

   Copyright (C) Intel, Inc. All rights reserved.
   Copyright (C) 2019-2023 Alexey V. Medvedev. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "async_suite.h"
#include "async_benchmark.h"
#include "async_workload.h"
#include "async_topology.h"

namespace async_suite {
    // Notified RMA (MPI 5.1, section 12.6): MPI_Put_notify moves the data and
    // bumps a notification counter at the target, so the target learns that
    // the data has arrived by polling a local counter -- no matching and no
    // separate synchronisation message.
    //
    // The receive buffer is the window itself, allocated with MPI_Win_allocate
    // so that shared-memory components (osc/sm) can serve it as well as
    // network ones (osc/ucx). Each source this rank receives from owns one
    // receive block and one notification counter here.
    class AsyncBenchmark_rma_notify_base : public AsyncBenchmark {
        public:
        MPI_Win win = MPI_WIN_NULL;
        // Blocks per rank in the receive buffer: the largest number of
        // sources over all ranks, so every rank has the same buffer layout.
        int nblocks = 0;
        actions_t comm_actions;
        // Per comm action: for a send, our block and counter index at the
        // target; for a receive, the source's block and counter index here.
        std::vector<int> slots;
        // Notifications seen so far per counter. Counters are never reset,
        // so waits compare against a cumulative expected value.
        std::vector<MPI_Count> expected;
        virtual size_t buf_size_multiplier_send() override { return 1; }
        virtual size_t buf_size_multiplier_recv() override { assert(nblocks); return nblocks; }
        virtual void init() override;
        virtual void finalize() override;
        MPI_Aint target_disp(int i, size_t b, int slot);
        void wait_notify(int slot);
    };

    class AsyncBenchmark_rma_notify : public AsyncBenchmark_rma_notify_base {
        public:
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        DEFINE_INHERITED(AsyncBenchmark_rma_notify, BenchmarkSuite<BS_GENERIC>);
    };

    class AsyncBenchmark_rma_inotify : public AsyncBenchmark_rma_notify_base {
        public:
        AsyncBenchmark_workload calc;
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        DEFINE_INHERITED(AsyncBenchmark_rma_inotify, BenchmarkSuite<BS_GENERIC>);
    };
}
