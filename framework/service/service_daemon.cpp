#include "framework/service/service_daemon.h"

namespace anakin {

namespace rpc {

void ServiceDaemon::operator()(std::function<int(int, int)> server_start,
                               std::vector<int> device_list,
                               int server_port) {
    // Our process ID and Session ID
    pid_t pid, sid;

    // Fork off the parent process
    pid = fork();

    if (pid < 0) {
        exit(EXIT_FAILURE);
    }

    // exit the parent process.
    if (pid > 0) {
        exit(EXIT_SUCCESS);
    }

    // Change the file mode mask, so we can use the files created by daemon.
    umask(0);

    // Create a new SID(a new session) for the child process
    sid = setsid();

    if (sid < 0) {
        // Log the failure
        exit(EXIT_FAILURE);
    }

    // Change the current working directory
    if ((chdir("/")) < 0) {
        exit(EXIT_FAILURE);
    }

    // Close out the standard file descriptors
    //close(STDIN_FILENO); // 0
    //close(STDOUT_FILENO); // 1
    //close(STDERR_FILENO); // 2

    // Daemon-specific initialization goes here */
    pid_t* pid_news = new pid_t[device_list.size()];

    for (;;) {
        for (auto dev_id : device_list) {
            if (!check_port_occupied(server_port) || !check_process_exist(pid_news[dev_id])) {
                LOG(WARNING) << " Create daemon process on device : " << dev_id;

                // reaped zombie process
                if (pid_news[dev_id]) {
                    waitpid(pid_news[dev_id], NULL, 0);
                }

                pid_news[dev_id] = fork();

                // fork new process
                if (pid_news[dev_id] == 0) {
                    prctl(PR_SET_NAME, "anakin_child_rpc_service");
                    int ret = server_start(server_port, dev_id);

                    if (ret == 0) {
                        exit(EXIT_SUCCESS);
                    } else {
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }

        sleep(30); // wait 30 seconds
    }

    exit(EXIT_SUCCESS);
}

bool ServiceDaemon::check_port_occupied(int port) {
    struct sockaddr_in client;
    int sk;

    client.sin_family = AF_INET;
    client.sin_port = htons(port);
    client.sin_addr.s_addr = inet_addr("0.0.0.0");

    sk = (int) socket(AF_INET, SOCK_STREAM, 0);

    int result = connect(sk, (struct sockaddr*) &client, sizeof(client));

    if (result == 0) {
        return true; // port is ocuupied.
    } else {
        return false;
    }
}

bool ServiceDaemon::check_process_exist(pid_t pid) {
    if (kill(pid, 0) == -1) {
        return false;
    } else {
        // process still exists
        return true;
    }
}

} /* namespace rpc */

} /* namespace anakin */

