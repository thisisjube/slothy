        start:
                                        // Instructions:    1
                                        // Expected cycles: 1
                                        // Expected IPC:    1.00
                                        //
                                        // Cycle bound:     1.0
                                        // IPC bound:       1.00
                                        //
                                        // Wall time:     0.02s
                                        // User time:     0.02s
                                        //
                                        // ----- cycle (expected) ------>
                                        // 0                        25
                                        // |------------------------|----
        vrgather.vv v8, v16, v24        // *.............................

                                         // ------ cycle (expected) ------>
                                         // 0                        25
                                         // |------------------------|-----
        // vrgather.vv v8, v16, v24      // *..............................

        end:
