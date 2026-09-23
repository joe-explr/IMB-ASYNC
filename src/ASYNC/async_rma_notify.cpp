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

#include <thread>
#include <algorithm>
#include <string>
#include "async_rma_notify.h"
#include "async_sys.h"
#include "async_average.h"
#include "async_topology.h"

namespace async_suite {
    void AsyncBenchmark_rma_notify_base::init() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        GET_PARAMETER(sys::host_alloc_t, host_alloc_mode);
        AsyncBenchmark::init();
        if (is_gpu) {
            throw std::runtime_error("AsyncBenchmark_rma_notify: GPU buffers are not supported");
        }
        topo = topohelper::create(p.get("rma_notify"), np, rank);
        comm_actions = topo->comm_actions();
        auto sources = topo->ranks_to_recv_from();
        for (auto &action : comm_actions) {
            int slot = -1;
            if (action.action == action_t::RECV) {
                slot = std::find(sources.begin(), sources.end(), action.rank) - sources.begin();
            } else if (action.action == action_t::SEND) {
                // Our slot at a target is our position among the target's
                // sources, so look at the topology from the target's side.
                auto target_sources = topo->clone(action.rank)->ranks_to_recv_from();
                auto it = std::find(target_sources.begin(), target_sources.end(), rank);
                if (it == target_sources.end()) {
                    throw std::runtime_error("AsyncBenchmark_rma_notify: topology is not symmetric");
                }
                slot = it - target_sources.begin();
            }
            slots.push_back(slot);
        }
        expected.assign(sources.size(), 0);
        int nsources = sources.size();
        MPI_Allreduce(&nsources, &nblocks, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        nblocks = std::max(nblocks, 1);
        AsyncBenchmark::alloc();

        // The window takes the place of the receive buffer alloc() made.
        // Window creation is collective, so inactive ranks join with an empty
        // window.
        sys::host_mem_free(host_rbuf, host_alloc_mode);
        host_rbuf = nullptr;
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "mpi_assert_max_num_notify", std::to_string(nsources).c_str());
        MPI_Win_allocate(allocated_size_recv, 1, info, MPI_COMM_WORLD, &host_rbuf, &win);
        MPI_Info_free(&info);
        MPI_Win_set_num_notify(win, MPI_INFO_NULL, nsources);
    }

    void AsyncBenchmark_rma_notify_base::finalize() {
        // host_rbuf is the window's memory, which MPI_Win_free releases.
        host_rbuf = nullptr;
        AsyncBenchmark::finalize();
        MPI_Win_free(&win);
    }

    // Every rank's receive buffer has the same layout, so where our block
    // sits in our own buffer is where it sits in the target's.
    MPI_Aint AsyncBenchmark_rma_notify_base::target_disp(int i, size_t b, int slot) {
        return get_rbuf(i, b, slot) - get_rbuf();
    }

    void AsyncBenchmark_rma_notify_base::wait_notify(int slot) {
        MPI_Count threshold = ++expected[slot];
        MPI_Count value = 0;
        do {
            MPI_Win_get_notify_value(win, slot, &value);
        } while (value < threshold);
    }

    void AsyncBenchmark_rma_notify::init() {
        AsyncBenchmark_rma_notify_base::init();
    }

    void AsyncBenchmark_rma_inotify::init() {
        AsyncBenchmark_rma_notify_base::init();
        calc.init();
    }

    bool AsyncBenchmark_rma_notify::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles,
                                              double &time, double &tover_comm, double &tover_calc) {
        tover_comm = 0;
        tover_calc = 0;
        if (!topo->is_active()) {
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        size_t b = get_send_bufsize_for_len(count);
        // get_sbuf() cycles through this many send buffers. A Put must be
        // locally complete before its buffer comes round again; nothing
        // else requires completing it at the origin, because the target
        // learns of the data from its counter.
        size_t nbufs = allocated_size_send / b;
        double t1 = 0, t2 = 0;
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
        for (int i = 0; i < ncycles + nwarmup; i++) {
            if (i == nwarmup) t1 = MPI_Wtime();
            for (size_t commstage = 0; commstage < comm_actions.size(); commstage++) {
                int rank = comm_actions[commstage].rank;
                int slot = slots[commstage];
                if (comm_actions[commstage].action == action_t::SEND) {
                    MPI_Put_notify(get_sbuf(i, b), count, datatype, rank, target_disp(i, b, slot),
                                   count, datatype, slot, win);
                } else if (comm_actions[commstage].action == action_t::RECV) {
                    wait_notify(slot);
                }
            }
            if ((i + 1) % nbufs == 0) {
                MPI_Win_flush_local_all(win);
            }
        }
        MPI_Win_flush_local_all(win);
        t2 = MPI_Wtime();
        MPI_Win_unlock_all(win);
        time = (t2 - t1) / ncycles;
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time, 0.0, 0.0, ncycles };
        return true;
    }

    bool AsyncBenchmark_rma_inotify::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles,
                                               double &time, double &tover_comm, double &tover_calc) {
        if (!topo->is_active()) {
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        size_t b = get_send_bufsize_for_len(count);
        double t1 = 0, t2 = 0, time_calc = 0, total_ctime = 0, total_tover_comm = 0, total_calc_slowdown_ratio = 0,
                                              local_ctime = 0, local_tover_comm = 0, local_calc_slowdown_ratio = 0;
        int nsends = std::count_if(comm_actions.begin(), comm_actions.end(),
                                   [](const peer_t &a) { return a.action == action_t::SEND; });
        if (nsends > MAX_REQUESTS_NUM) {
            throw std::runtime_error("AsyncBenchmark_rma_inotify: MAX_REQUESTS_NUM is too little for the topology");
        }
        MPI_Request *requests = (MPI_Request *)calloc(sizeof(MPI_Request), std::max(nsends, 1));
        calc.reqs = requests;
        calc.num_requests = nsends;
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
        for (int i = 0; i < ncycles + nwarmup; i++) {
            if (i == nwarmup) t1 = MPI_Wtime();
            int nreq = 0;
            for (size_t commstage = 0; commstage < comm_actions.size(); commstage++) {
                int rank = comm_actions[commstage].rank;
                int slot = slots[commstage];
                if (comm_actions[commstage].action == action_t::SEND) {
                    MPI_Rput_notify(get_sbuf(i, b), count, datatype, rank, target_disp(i, b, slot),
                                    count, datatype, slot, win, &requests[nreq++]);
                }
            }
            calc.benchmark(count, datatype, 0, 1, local_ctime, local_tover_comm, local_calc_slowdown_ratio);
            MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
            // Nothing is posted for receives: the data is in place once the
            // counter says so.
            for (size_t commstage = 0; commstage < comm_actions.size(); commstage++) {
                if (comm_actions[commstage].action == action_t::RECV) {
                    wait_notify(slots[commstage]);
                }
            }
            if (i >= nwarmup) {
                total_ctime += local_ctime;
                total_tover_comm += local_tover_comm;
                total_calc_slowdown_ratio += local_calc_slowdown_ratio;
            }
        }
        t2 = MPI_Wtime();
        MPI_Win_unlock_all(win);
        time = (t2 - t1) / ncycles;
        time_calc = total_ctime / ncycles;
        tover_comm = total_tover_comm / ncycles;
        double calc_slowdown_ratio = total_calc_slowdown_ratio / ncycles;
        double time_comm = time - time_calc;
        tover_calc = time_comm * calc_slowdown_ratio;
        MPI_Barrier(MPI_COMM_WORLD);
        free(requests);
        results[count] = result { true, time, time_comm + tover_comm, tover_calc, ncycles };
        return true;
    }

    DECLARE_INHERITED(AsyncBenchmark_rma_notify, sync_rma_notify)
    DECLARE_INHERITED(AsyncBenchmark_rma_inotify, async_rma_notify)
}
