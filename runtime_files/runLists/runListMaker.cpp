void runListMaker(){

    std::string dir = "/Volumes/PortableSSD/pp200_production_2012/2024-03-12/Events";

    std::string badRunsList = "/Users/tanmaypani/star-workspace/preprocessing/data/runLists/pp200_production_2012_BAD_Issac.list";
    std::ifstream badRunsListFile(badRunsList);
    std::string _badRunsLine;
    std::set<std::string> _badRunList; 
    while(std::getline(badRunsListFile, _badRunsLine)){
        _badRunList.insert(_badRunsLine);
    }
    badRunsListFile.close();
    cout<<_badRunList.size()<<endl;

    std::set<std::string> _allRunList;
    for (auto& file : std::filesystem::directory_iterator(dir)){
        std::string fileName = file.path().filename().string();
        char* fileNameC_str = fileName.data();
        char* token = std::strtok(fileNameC_str, "_");
        std::vector<std::string> tokens = {};
        while(token){
            tokens.emplace_back(token);
            token = std::strtok(nullptr, "_");
        }
        //cout<<std::quoted(tokens[0])<<endl;
        _allRunList.insert(tokens[0]);
    }

    cout<<_allRunList.size()<<endl;

    for(auto _badRun : _badRunList){
        _allRunList.erase(_badRun);
    }

    cout<<_allRunList.size()<<endl;

    std::ofstream out("pp200_production_2012_goodRuns.list");
    auto _nRunsToWrite = _allRunList.size();
    for(auto _run : _allRunList){
        out << _run ;
        if(--_nRunsToWrite != 0)out << std::endl;
    }
    out.close();

}