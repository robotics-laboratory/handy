#include <opencv2/core/core.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/aruco.hpp>

int main() {
    // cv::aruco::Dictionary dictionary =
    // cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250); auto charuco_board =
    // cv::aruco::CharucoBoard(cv::Size{10, 7}, 0.06f, 0.04f, dictionary);
    // charuco_board.setLegacyPattern(true);

    // cv::Mat image;
    // charuco_board.generateImage(cv::Size{560, 410}, image);
    // cv::imwrite("board.png", image);

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::GridBoard board(cv::Size{7, 5}, 0.04f, 0.01f, dictionary);
    cv::Mat image;
    board.generateImage(cv::Size{1920, 1400}, image);
    cv::imwrite("board_aruco.png", image);
    return 0;
}