// @ts-check
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';
import starlightThemeNova from 'starlight-theme-nova';

export default defineConfig({
  site: 'https://golden.retriever.build',
  integrations: [
    starlight({
      title: 'GoldenRetriever',
      description: 'Applied Retriever examples, robot payload references, visualization lanes, and Hub pack candidates.',
      logo: {
        src: './src/assets/retriever-illustrative.jpeg',
        alt: 'GoldenRetriever',
      },
      favicon: '/assets/logo.svg',
      customCss: ['./src/styles/golden.css'],
      social: [
        { icon: 'open-book', label: 'Retriever core docs', href: 'https://retriever.build/' },
        { icon: 'external', label: 'Retriever project home', href: 'https://openretriever.org/' },
        { icon: 'github', label: 'GoldenRetriever source on GitHub', href: 'https://github.com/openretriever/golden-retriever' },
      ],
      plugins: [
        starlightThemeNova({
          nav: [
            { label: 'Core docs', href: 'https://retriever.build/' },
            { label: 'Project home', href: 'https://openretriever.org/' },
          ],
        }),
      ],
      components: {
        PageTitle: './src/components/PageTitleWithSource.astro',
      },
      sidebar: [
        {
          label: 'Start',
          items: [
            { label: 'Overview', link: '/' },
            { label: 'Hub Pack Quickstart', slug: 'examples/golden-hub-proof' },
            { label: 'Example Catalog', slug: 'examples' },
          ],
        },
        {
          label: 'Run Examples',
          items: [
            { label: 'Perception and Memory', slug: 'examples/perception-memory' },
            { label: 'Language and Grounding', slug: 'examples/language-grounding' },
            { label: 'Pipeline Composition', slug: 'examples/pipeline-composition' },
            { label: 'Simulation and Visualization', slug: 'examples/simulation-visualization' },
          ],
        },
        {
          label: 'Notebooks',
          items: [
            { label: 'Language → Plan', slug: 'notebooks/language-caption-plan' },
          ],
        },
        {
          label: 'Reuse Robot Payloads',
          items: [
            { label: 'Payload Overview', slug: 'robot-payloads' },
            { label: 'Choose a Payload', slug: 'robot-payloads/type-catalog' },
            { label: 'Flow I/O Contracts', slug: 'robot-payloads/flow-contracts' },
            { label: 'Data and Event Streams', slug: 'robot-payloads/data-event-streams' },
            { label: 'LeRobot Export', slug: 'robot-payloads/lerobot-export' },
          ],
        },
        {
          label: 'Promote to Hub Packs',
          items: [
            { label: 'Hub Packs', slug: 'hub' },
            { label: 'Export Catalog', slug: 'hub/export-catalog' },
            { label: 'Maturity Guide', slug: 'hub/pack-roadmap' },
          ],
        },
      ],
    }),
  ],
});
