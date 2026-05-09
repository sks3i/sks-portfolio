// @ts-check
import { defineConfig } from 'astro/config';

const repo = process.env.GITHUB_REPOSITORY?.split('/')[1] ?? '';
const isGitHubActions = process.env.GITHUB_ACTIONS === 'true';
const isUserSiteRepo = repo.endsWith('.github.io');
const base = isGitHubActions && repo && !isUserSiteRepo ? `/${repo}` : '/';
const site = process.env.GITHUB_REPOSITORY_OWNER
	? `https://${process.env.GITHUB_REPOSITORY_OWNER}.github.io`
	: undefined;

export default defineConfig({
	site,
	base,
});
