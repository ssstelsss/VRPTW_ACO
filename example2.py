from vrptw_base import VrptwGraph
from basic_aco import BasicACO


if __name__ == '__main__':
    file_path = './data/test.txt'
    ants_num = 10
    max_iter = 200
    beta = 2
    q0 = 0.1
    show_figure = False

    graph = VrptwGraph(file_path)
    print(graph)
    basic_aco = BasicACO(graph, ants_num=ants_num, max_iter=max_iter, beta=beta, q0=q0,
                         whether_or_not_to_show_figure=show_figure)

    basic_aco.run_basic_aco()
