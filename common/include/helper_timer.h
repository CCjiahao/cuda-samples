#ifndef __HELPER_TIMER_H__
#define __HELPER_TIMER_H__

#include <vector>
#include <chrono> 


class StopWatch {
private:
    std::chrono::system_clock::time_point start_time;
    float diff_time;
    float total_time;
    bool running;
    int clock_sessions;

private:
    inline float getDiffTime() {
        std::chrono::system_clock::time_point t_time = std::chrono::system_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(t_time - start_time);
        return duration.count() * 0.001;
    }

public:
    StopWatch() : start_time(), diff_time(0.0), total_time(0.0), running(false), clock_sessions(0) {}
    virtual ~StopWatch() {}

public:
    inline void start() {
        start_time = std::chrono::system_clock::now();
        running = true;
    }

    inline void stop() {
        diff_time = getDiffTime();
        total_time += diff_time;
        running = false;
        clock_sessions++;
    }

    inline void reset() {
        diff_time = total_time = clock_sessions = 0;
        if (running) {
            start_time = std::chrono::system_clock::now();
        }
    }

    inline float getTime() {
        float retval = total_time;
        if (running) {
            retval += getDiffTime();
        }
        return retval;
    }

    inline float getAverageTime() {
        return (clock_sessions > 0) ? (total_time / clock_sessions) : 0.0f;
    }
};

#endif  // __HELPER_TIMER_H__
