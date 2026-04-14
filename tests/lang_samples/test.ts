interface User {
    name: string;
    score: number;
    active: boolean;
}

type ScoreMap = Map<string, number>;

function rankUsers<T extends User>(users: T[]): T[] {
    return users
        .filter(u => u.active)
        .sort((a, b) => b.score - a.score);
}

const getUser = (id: number): any => {
    // @ts-ignore
    return globalData[id];
};

async function fetchScores(url: string): Promise<ScoreMap> {
    const response = await fetch(url);
    const data: User[] = await response.json();
    const map: ScoreMap = new Map();
    for (const user of data) {
        map.set(user.name, user.score);
    }
    return map;
}
